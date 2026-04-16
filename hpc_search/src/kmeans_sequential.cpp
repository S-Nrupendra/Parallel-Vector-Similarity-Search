#include "fingerprint_db.h"
#include <vector>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
double elapsed_ms_k(Clock::time_point start){
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Tanimoto similarity
static float tanimoto_seq(const uint64_t* a, const uint64_t* b){
    int common = 0, total = 0;
    for (int i = 0; i < FP_WORDS; i++){
        common += __builtin_popcountll(a[i] & b[i]);
        total += __builtin_popcountll(a[i] | b[i]);
    }
    return total == 0 ? 1.0f : (float)common / total;
}

// Majority vote: recompute centroid from assigned molecules
static void update_centroid(
    Centroid& centroid,
    const FingerprintDB& db,
    const std::vector<int>& assignments,
    int cluster_id,
    int N)
{
    // Count votes per bit position
    // Each fingerprint has FP_WORDS * 64 = 2048 bits
    std::vector<int> bit_counts(FP_BITS, 0);
    int count = 0;

    for(int i = 0; i < N; i++){
        if(assignments[i] != cluster_id) continue;
        count++;
        // For each word, check each bit
        for(int w = 0; w < FP_WORDS; w++){
            uint64_t word = db.molecules[i].fp[w];
            for(int b = 0; b < 64; b++){
                if((word >> b) & 1ULL)
                    bit_counts[w * 64 + b]++;
            }
        }
    }

    if (count == 0) return; // empty cluster — keep old centroid

    // Majority vote → set centroid bits
    memset(centroid.fp, 0, sizeof(centroid.fp));
    for(int w = 0; w < FP_WORDS; w++){
        for(int b = 0; b < 64; b++){
            if(bit_counts[w * 64 + b] * 2 > count)
                centroid.fp[w] |= (1ULL << b);
        }
    }
}

KMeansResult kmeans_sequential(const FingerprintDB& db, int K, int max_iters){
    int N = db.size();
    auto total_start = Clock::now();

    // Initialize centroids by picking K random molecules
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
        for(int i = 0; i < N; i++){
            float best_sim = -1.0f;
            int best_k = 0;
            for(int k = 0; k < K; k++){
                float sim = tanimoto_seq(db.molecules[i].fp, centroids[k].fp);
                if(sim > best_sim){
                    best_sim = sim;
                    best_k = k;
                }
            }
            new_assignments[i] = best_k;
        }

        double iter_ms = elapsed_ms_k(iter_start);
        std::cout<<"  [Seq] Iteration "<< iter + 1<< ": "<<iter_ms<<" ms\n";

        // check for convergence
        if(new_assignments == assignments){
            std::cout<<"  [Seq] Converged at iteration "<<iter + 1<<"\n";
            assignments = new_assignments;
            break;
        }
        assignments = new_assignments;

        // update step
        for(int k = 0; k < K; k++)
            update_centroid(centroids[k], db, assignments, k, N);
    }

    KMeansResult result;
    result.assignments = assignments;
    result.centroids = centroids;
    result.iterations = iterations;
    result.total_ms = elapsed_ms_k(total_start);
    return result;
}