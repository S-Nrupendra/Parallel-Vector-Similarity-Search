#include "fingerprint_db.h"
#include <immintrin.h>  // AVX2 intrinsics
#include <omp.h>
#include <queue>
#include <vector>
#include <algorithm>

// AVX2 + POPCNT accelerated tanimoto
// Processes 4 x uint64 words at a time using AVX2
static float tanimoto_simd(const uint64_t* a, const uint64_t* b) {
    int common = 0, total = 0;

    for (int i = 0; i < FP_WORDS; i += 4) {
        __m256i va = _mm256_loadu_si256((__m256i*)(&a[i]));
        __m256i vb = _mm256_loadu_si256((__m256i*)(&b[i]));

        __m256i vand = _mm256_and_si256(va, vb);
        __m256i vor  = _mm256_or_si256(va, vb);

        // Use union to extract uint64 values safely
        union { __m256i v; uint64_t u[4]; } and_u, or_u;
        and_u.v = vand;
        or_u.v  = vor;

        common += __builtin_popcountll(and_u.u[0]);
        common += __builtin_popcountll(and_u.u[1]);
        common += __builtin_popcountll(and_u.u[2]);
        common += __builtin_popcountll(and_u.u[3]);

        total  += __builtin_popcountll(or_u.u[0]);
        total  += __builtin_popcountll(or_u.u[1]);
        total  += __builtin_popcountll(or_u.u[2]);
        total  += __builtin_popcountll(or_u.u[3]);
    }

    return total == 0 ? 1.0f : (float)common / total;
}

std::vector<SearchResult> search_simd(
    const FingerprintDB& db,
    const Molecule& query,
    int K)
{
    int N = db.size();
    int num_threads = omp_get_max_threads();

    using Pair = std::pair<float, int>;
    using MinHeap = std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>;

    // One local heap per thread
    std::vector<MinHeap> local_heaps(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        MinHeap& local = local_heaps[tid];

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            float sim = tanimoto_simd(query.fp, db.molecules[i].fp);
            if ((int)local.size() < K) {
                local.push({sim, i});
            } else if (sim > local.top().first) {
                local.pop();
                local.push({sim, i});
            }
        }
    }

    // Merge all local heaps into final heap
    MinHeap final_heap;
    for (auto& heap : local_heaps) {
        while (!heap.empty()) {
            float sim = heap.top().first;
            int idx   = heap.top().second;
            heap.pop();
            if ((int)final_heap.size() < K) {
                final_heap.push({sim, idx});
            } else if (sim > final_heap.top().first) {
                final_heap.pop();
                final_heap.push({sim, idx});
            }
        }
    }


    // Extract results
    std::vector<SearchResult> results;
    while (!final_heap.empty()) {
        float sim = final_heap.top().first;
        int idx   = final_heap.top().second;
        final_heap.pop();
        results.push_back({idx, sim});
    }
    std::reverse(results.begin(), results.end());
    return results;
}