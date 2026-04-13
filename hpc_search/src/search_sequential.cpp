/*
This is the slow but correct search.
It checks every single molecule one by one, computes the Tanimoto similarity, and keeps track of the top-K results using a min-heap.

This is Tanimoto similarity. For each of the 32 uint64 words:
    -> a[i] & b[i] → bits present in both molecules
    -> a[i] | b[i] → bits present in either molecule
    -> popcount counts how many 1-bits are set
    -> Result: how much overlap / total = similarity score (0 to 1)

We use a min-heap to maintain the top-K results.
If we have fewer than K results, we just add.
Once we have K, we only add if the new similarity is better than the worst in the heap.
This avoids sorting the entire list of 2.4M molecules, which would be much slower.
*/
#include "fingerprint_db.h"
#include <queue>
#include <vector>
#include <algorithm>

// Tanimoto similarity between two fingerprints
static float tanimoto(const uint64_t* a, const uint64_t* b) {
    int common = 0, total = 0;
    for (int i = 0; i < FP_WORDS; i++) {
        common += __builtin_popcountll(a[i] & b[i]);
        total  += __builtin_popcountll(a[i] | b[i]);
    }
    return total == 0 ? 1.0f : (float)common / total;
}

// Returns top-K most similar molecules (sequential)
std::vector<SearchResult> search_sequential(
    const FingerprintDB& db,
    const Molecule& query,
    int K)
{
    // Min-heap: keeps track of top-K
    // pair<similarity, index> — min on top so we can drop lowest easily
    using Pair = std::pair<float, int>;
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;

    int N = db.size();
    for (int i = 0; i < N; i++) {
        float sim = tanimoto(query.fp, db.molecules[i].fp);
        if ((int)heap.size() < K) {
            heap.push({sim, i});
        } else if (sim > heap.top().first) {
            heap.pop();
            heap.push({sim, i});
        }
    }

    // Extract results (will be in ascending order)
    std::vector<SearchResult> results;
    while (!heap.empty()) {
        float sim = heap.top().first;
        int idx   = heap.top().second;
        heap.pop();
        results.push_back({idx, sim});
    }
    // Reverse to get descending order
    std::reverse(results.begin(), results.end());
    return results;
}