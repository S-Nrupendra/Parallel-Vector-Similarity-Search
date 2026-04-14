#include "fingerprint_db.h"
#include <omp.h>
#include <queue>
#include <vector>
#include <algorithm>

// Same tanimoto as sequential
static float tanimoto(const uint64_t* a, const uint64_t* b) {
    int common = 0, total = 0;
    for (int i = 0; i < FP_WORDS; i++) {
        common += __builtin_popcountll(a[i] & b[i]);
        total  += __builtin_popcountll(a[i] | b[i]);
    }
    return total == 0 ? 1.0f : (float)common / total;
}

std::vector<SearchResult> search_openmp(
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
        for(int i = 0;i < N;i++){
            float sim = tanimoto(query.fp, db.molecules[i].fp);
            if((int)local.size() < K){
                local.push({sim, i});
            }
            else if(sim > local.top().first){
                local.pop();
                local.push({sim, i});
            }
        }
    }

    // Merge all local heaps into one final heap
    MinHeap final_heap;
    for(auto& heap : local_heaps){
        while(!heap.empty()){
            float sim = heap.top().first;
            int idx   = heap.top().second;
            heap.pop();
            
            if((int)final_heap.size() < K){
                final_heap.push({sim, idx});
            }
            else if(sim > final_heap.top().first){
                final_heap.pop();
                final_heap.push({sim, idx});
            }
        }
    }

    // Extract results
    std::vector<SearchResult> results;
    while(!final_heap.empty()){
        float sim = final_heap.top().first;
        int idx   = final_heap.top().second;
        final_heap.pop();

        results.push_back({idx, sim});
    }

    std::reverse(results.begin(), results.end());
    return results;
}