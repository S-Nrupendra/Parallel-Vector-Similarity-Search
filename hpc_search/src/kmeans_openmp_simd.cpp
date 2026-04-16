#include "fingerprint_db.h"
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
double elapsed_ms_komp(Clock::time_point start){
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// AVX2 tanimoto (same as search_simd.cpp)
static float tanimoto_simd(const uint64_t* a, const uint64_t* b){
    int common = 0, total = 0;
    for(int i = 0; i < FP_WORDS; i += 4){
        __m256i va = _mm256_loadu_si256((__m256i*)(&a[i]));
        __m256i vb = _mm256_loadu_si256((__m256i*)(&b[i]));

        __m256i vand = _mm256_and_si256(va, vb);
        __m256i vor  = _mm256_or_si256(va, vb);

        union { __m256i v; uint64_t u[4]; } and_u, or_u;
        and_u.v = vand;
        or_u.v  = vor;

        common += __builtin_popcountll(and_u.u[0]);
        common += __builtin_popcountll(and_u.u[1]);
        common += __builtin_popcountll(and_u.u[2]);
        common += __builtin_popcountll(and_u.u[3]);

        total += __builtin_popcountll(or_u.u[0]);
        total += __builtin_popcountll(or_u.u[1]);
        total += __builtin_popcountll(or_u.u[2]);
        total += __builtin_popcountll(or_u.u[3]);
    }
    return total == 0 ? 1.0f : (float)common / total;
}

// Parallel majority vote centroid update
static void update_centroid_parallel(
    Centroid& centroid,
    const FingerprintDB& db,
    const std::vector<int>& assignments,
    int cluster_id,
    int N)
{
    std::vector<int> bit_counts(FP_BITS, 0);
    int count = 0;

    #pragma omp parallel for reduction(+:count) schedule(static)
    for(int i = 0; i < N; i++){
        if(assignments[i] != cluster_id) continue;
        #pragma omp atomic
        count++;
        for(int w = 0; w < FP_WORDS; w++){
            uint64_t word = db.molecules[i].fp[w];
            for(int b = 0; b < 64; b++){
                if((word >> b) & 1ULL){
                    #pragma omp atomic
                    bit_counts[w * 64 + b]++;
                }
            }
        }
    }

    if(count == 0) return;

    memset(centroid.fp, 0, sizeof(centroid.fp));
    for(int w = 0; w < FP_WORDS; w++)
        for(int b = 0; b < 64; b++)
            if(bit_counts[w * 64 + b] * 2 > count)
                centroid.fp[w] |= (1ULL << b);
}

KMeansResult kmeans_openmp_simd(const FingerprintDB& db, int K, int max_iters){
    int N = db.size();
    auto total_start = Clock::now();

    // Same initial centroids as sequential
    std::vector<Centroid> centroids(K);
    srand(42);
    for(int k = 0; k < K; k++){
        int idx = rand() % N;
        memcpy(centroids[k].fp, db.molecules[idx].fp, sizeof(Centroid));
    }

    std::vector<int> assignments(N, -1);
    int iterations = 0;

    for(int iter = 0; iter < max_iters; iter++){
        auto iter_start = Clock::now();
        iterations++;

        std::vector<int> new_assignments(N);

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < N; i++){
            float best_sim = -1.0f;
            int   best_k   = 0;
            for(int k = 0; k < K; k++){
                float sim = tanimoto_simd(db.molecules[i].fp, centroids[k].fp);
                if(sim > best_sim){
                    best_sim = sim;
                    best_k   = k;
                }
            }
            new_assignments[i] = best_k;
        }

        double iter_ms = elapsed_ms_komp(iter_start);
        std::cout<<"  [OMP+SIMD] Iteration "<<iter + 1<< ": "<<iter_ms<<" ms\n";

        if(new_assignments == assignments){
            std::cout<<"  [OMP+SIMD] Converged at iteration "<<iter + 1<<"\n";
            assignments = new_assignments;
            break;
        }
        assignments = new_assignments;

        for(int k = 0; k < K; k++)
            update_centroid_parallel(centroids[k], db, assignments, k, N);
    }

    KMeansResult result;
    result.assignments = assignments;
    result.centroids = centroids;
    result.iterations = iterations;
    result.total_ms = elapsed_ms_komp(total_start);
    return result;
}