#include "fingerprint_db.h"
#include <iostream>
#include <chrono>
#include <fstream>

void save_assignments(const KMeansResult& result, const std::string& path){
    std::ofstream f(path);
    f<<"molecule_index,cluster_id\n";
    for(int i = 0;i < (int)result.assignments.size();i++)
        f<<i<<","<<result.assignments[i]<<"\n";
    std::cout<<"Assignments saved to "<<path<<"\n";
}

int main(){
    const std::string bin_path = "../data/chembl_fingerprints.bin";
    const int K = 100;
    const int MAX_ITERS = 20;

    // Load database
    std::cout<<"=== Loading Database ===\n";
    FingerprintDB db = load_database(bin_path);
    std::cout<<"\n";

    // Sequential KMeans
    std::cout<< "=== Sequential KMeans (K="<<K<<", max_iters="<<MAX_ITERS<<") ===\n";
    KMeansResult seq_result = kmeans_sequential(db, K, MAX_ITERS);
    std::cout<<"Total time : "<<seq_result.total_ms<<" ms\n";
    std::cout<<"Iterations : "<<seq_result.iterations<<"\n\n";

    // OpenMP + SIMD KMeans
    std::cout<<"=== OpenMP+SIMD KMeans (K="<<K<<", max_iters="<<MAX_ITERS<<") ===\n";
    KMeansResult omp_result = kmeans_openmp_simd(db, K, MAX_ITERS);
    std::cout<<"Total time : "<<omp_result.total_ms<<" ms\n";
    std::cout<<"Iterations : "<<omp_result.iterations<<"\n\n";

    // Benchmark summary
    double speedup = seq_result.total_ms / omp_result.total_ms;
    std::cout<<"=== KMeans Benchmark Summary ===\n";
    std::cout<<"Sequential   : "<<seq_result.total_ms<<" ms  (1.00x)\n";
    std::cout<<"OpenMP+SIMD  : "<<omp_result.total_ms<<" ms  ("<<speedup<<"x)\n";

    save_assignments(omp_result, "../data/assignments.csv");

    return 0;
}