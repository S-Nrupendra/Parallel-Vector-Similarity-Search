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

// Top-K result
struct SearchResult {
    int   index;        // which molecule in the DB
    float similarity;   // tanimoto score (0 to 1)
};

std::vector<SearchResult> search_openmp(const FingerprintDB&, const Molecule&, int K);
std::vector<SearchResult> search_simd(const FingerprintDB&, const Molecule&, int K);