/*
This is where the Program starts. It:
    -> Calls load_database() to read the binary file and populate the FingerprintDB structure.
    -> Uses the first molecule in the database as a query.
    -> Calls search_sequential() to find the top K most similar molecules based on Tanimoto similarity.
    -> Measures and prints the time taken for loading and searching, as well as the top K
*/
#include "fingerprint_db.h"
#include <iostream>
#include <chrono>
#include <vector>

// Declaration
FingerprintDB load_database(const std::string& path);
std::vector<SearchResult> search_sequential(const FingerprintDB&, const Molecule&, int);

// Time helper
using Clock = std::chrono::high_resolution_clock;
double elapsed_ms(Clock::time_point start){
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void print_results(const std::vector<SearchResult>& results){
    for(auto& r : results){
        std::cout << "  mol[" << r.index << "] similarity = " << r.similarity << "\n";   
    }
}

int main(){
    const std::string bin_path = "../data/chembl_fingerprints.bin";
    const int K = 5;

    // Load database
    std::cout<< "=== Loading Database ===\n";
    auto t0 = Clock::now();
    FingerprintDB db = load_database(bin_path);
    std::cout<<"Load time: "<<elapsed_ms(t0)<<" ms\n\n";

    // Use molecule 0 as query
    Molecule query = db.molecules[0];

    // Sequential search
    std::cout << "=== Sequential Search ===\n";
    auto t1 = Clock::now();
    auto results_seq = search_sequential(db, query, K);
    double seq_ms = elapsed_ms(t1);
    std::cout << "Time: " << seq_ms << " ms\n";
    print_results(results_seq);

    // OpenMP search
    std::cout<<"\n=== OpenMP Search ===\n";
    auto t2 = Clock::now();
    auto results_omp = search_openmp(db, query, K);
    double omp_ms = elapsed_ms(t2);
    std::cout<<"Time : " <<omp_ms<<" ms\n";
    print_results(results_omp);

    // SIMD search
    std::cout << "\n=== SIMD + OpenMP Search ===\n";
    auto t3 = Clock::now();
    auto results_simd = search_simd(db, query, K);
    double simd_ms = elapsed_ms(t3);
    std::cout << "Time: " << simd_ms << " ms\n";
    print_results(results_simd);

    // Correctness checks
    std::cout << "\n=== Correctness Check ===\n";
    bool omp_ok = true, simd_ok = true;
    for (int i = 0; i < K; i++) {
        if (results_seq[i].index != results_omp[i].index)  omp_ok  = false;
        if (results_seq[i].index != results_simd[i].index) simd_ok = false;
    }
    std::cout << "Sequential vs OpenMP     : " << (omp_ok  ? "MATCH ✓" : "MISMATCH ✗") << "\n";
    std::cout << "Sequential vs SIMD+OpenMP: " << (simd_ok ? "MATCH ✓" : "MISMATCH ✗") << "\n";
    
    // Benchmark summary
    std::cout << "\n=== Benchmark Summary ===\n";
    std::cout << "Sequential    : " << seq_ms  << " ms  (1.00x)\n";
    std::cout << "OpenMP        : " << omp_ms  << " ms  (" << seq_ms/omp_ms  << "x)\n";
    std::cout << "SIMD + OpenMP : " << simd_ms << " ms  (" << seq_ms/simd_ms << "x)\n";

    return 0;
}