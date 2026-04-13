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
    std::cout<< "=== Sequential Search ===\n";
    auto t1 = Clock::now();
    auto results = search_sequential(db, query, K);
    double seq_ms = elapsed_ms(t1);
    std::cout<<"Time: "<<seq_ms<<" ms\n";
    std::cout<<"Top-"<<K<<" results:\n";
    for(auto& r : results)
        std::cout<<" mol["<<r.index<<"] similarity = "<<r.similarity<<"\n";
    
    std::cout<<"\n=== Benchmark Summary ===\n";
    std::cout<<"Sequential : "<<seq_ms<<" ms (1.00x)\n";

    return 0;
}