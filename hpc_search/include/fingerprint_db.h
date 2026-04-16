/*
include/fingerprint_db.h - Shared Data Structures
This is the "blueprint" file Every other file includes this

In preprocessing — each molecule's fingerprint is 2048 bits.
We store those 2048 bits as 32 chunks of 64 bits each (uint64_t).
This is because CPU instructions work on 64-bit integers natively.
*/
#pragma once
#include<cstdint>
#include<vector>
#include<string>

// Each fingerprint = 2048 bits = 32 x uint64
constexpr int FP_WORDS = 32;
constexpr int FP_BITS  = 2048;

// One molecule
struct Molecule {
    uint64_t fp[FP_WORDS]; // one molecule = array of 32 uint64
};

// The entire database
struct FingerprintDB {
    std::vector<Molecule> molecules; // all 2.4M molecules
    int size() const { return (int)molecules.size(); }
};

// Top-K result // Search result for one molecule in the DB
struct SearchResult {
    int   index;        // which molecule in the DB
    float similarity;   // tanimoto score (0 to 1)
};

// KMean centroid - same shape as a molecule
struct Centroid{
    uint64_t fp[FP_WORDS];
};

// KMean result
struct KMeansResult{
    std::vector<int> assignments; // assigments[i] = cluster id of molecule i
    std::vector<Centroid> centroids; // final K centroids
    int iterations; // how many iterations ran
    double total_ms; // total time taken
};

FingerprintDB load_database(const std::string& path);

std::vector<SearchResult> search_sequential(const FingerprintDB&, const Molecule&, int K);
std::vector<SearchResult> search_openmp(const FingerprintDB&, const Molecule&, int K);
std::vector<SearchResult> search_simd(const FingerprintDB&, const Molecule&, int K);

// KMeans
KMeansResult kmeans_sequential(const FingerprintDB&, int K, int max_iters);
KMeansResult kmeans_openmp_simd(const FingerprintDB&, int K, int max_iters);