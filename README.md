# Parallel Vector Similarity Search Engine
### High Performance Computing Project | ChEMBL Chemistry Dataset

---

## Overview

A high-performance search engine for molecular fingerprint vectors. Given a database of 2.85 million molecules, the system finds the **K most chemically similar molecules** to a query using Tanimoto similarity — the same core operation that powers modern vector databases like Pinecone and Milvus in AI/RAG pipelines.

The project demonstrates how HPC techniques (OpenMP, SIMD/AVX2) can dramatically accelerate brute-force search and clustering over large-scale binary vector databases.

---

## Industry Relevance

Vector similarity search is the backbone of modern AI:
- **Retrieval-Augmented Generation (RAG)** — when LLMs like ChatGPT "look up" relevant documents, vector search happens under the hood
- **Drug discovery** — finding chemically similar molecules to a drug candidate
- **Vector databases** (Pinecone, Milvus, FAISS) — all built on fast nearest-neighbor search

---

## Dataset

**Source:** [ChEMBL Database](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)

| Property | Value |
|---|---|
| Total molecules | 2,854,639 |
| Fingerprint type | Morgan Fingerprints (radius=2) |
| Fingerprint size | 2048 bits |
| C++ representation | `uint64_t[32]` per molecule |
| Binary file size | 730.8 MB |

Each molecule is represented as a **2048-bit Morgan fingerprint** — a binary vector where each bit represents the presence or absence of a particular chemical substructure.

---

## Distance Metric — Tanimoto Similarity

Instead of Euclidean distance, molecular similarity uses **Tanimoto similarity** (Jaccard on bit vectors):

```
T(A, B) = popcount(A AND B) / popcount(A OR B)
```

- Score of **1.0** = identical molecules
- Score of **0.0** = completely different molecules
- Uses only bitwise AND, OR, and popcount — single CPU instructions, no floating point

---

## Project Structure

```
hpc_search/
├── data/
│   ├── chembl_fingerprints.bin    ← packed binary fingerprints (730 MB)
│   └── assignments.csv            ← KMeans cluster assignments output
├── include/
│   └── fingerprint_db.h           ← shared data structures & declarations
├── src/
│   ├── io.cpp                     ← binary file loader
│   ├── main.cpp                   ← search benchmark entry point
│   ├── search_sequential.cpp      ← brute force single-thread search
│   ├── search_openmp.cpp          ← OpenMP parallel search
│   ├── search_simd.cpp            ← AVX2 + OpenMP search
│   ├── kmeans_main.cpp            ← KMeans benchmark entry point
│   ├── kmeans_sequential.cpp      ← single-thread KMeans
│   └── kmeans_openmp_simd.cpp     ← OpenMP + AVX2 KMeans
├── hpc-eda.ipynb                  ← preprocessing + visualization notebook
└── CMakeLists.txt
```

Two separate executables:
- **`./hpc_search`** — similarity search benchmark
- **`./hpc_kmeans`** — clustering benchmark

---

## Step 0 — Data Preprocessing

Run `hpc-eda.ipynb` on Kaggle to:
1. Load ChEMBL `.txt` file, extract `chembl_id` and `canonical_smiles`
2. Drop nulls, duplicates, and invalid SMILES
3. Generate Morgan fingerprints using RDKit
4. Pack 2048-bit fingerprints into `uint64` format
5. Save `chembl_fingerprints.bin` for C++ consumption

**Why uint64 packing?**

RDKit outputs each bit as a separate `uint8` (2048 bytes/molecule — wasteful). We pack 64 bits into each `uint64`, reducing storage to 256 bytes/molecule and enabling AVX2 to process 4 words simultaneously.

```
RDKit output  : 2048 × uint8  = 2048 bytes per molecule
After packing :   32 × uint64 =  256 bytes per molecule  (8x smaller)
```

---

## Step 1 — I/O & Architecture

**Core data structures (`fingerprint_db.h`):**

```cpp
constexpr int FP_WORDS = 32;   // 32 × 64-bit = 2048 bits

struct Molecule      { uint64_t fp[FP_WORDS]; };
struct FingerprintDB { std::vector<Molecule> molecules; };
struct SearchResult  { int index; float similarity; };
struct Centroid      { uint64_t fp[FP_WORDS]; };
```

The binary file is read in a **single `fread` call** — no loops, no parsing, direct memory map into the struct array. Load time: ~11 seconds (WSL filesystem overhead on 730 MB).

**Build flags:**
```
-O2       → compiler optimizations
-mavx2    → enable AVX2 instructions
-mpopcnt  → hardware POPCNT instruction
-fopenmp  → enable OpenMP threading
```

---

## Step 2 — Sequential Search (Baseline)

Brute-force: compute Tanimoto between query and every molecule one at a time. Maintains a **min-heap of size K** to track top-K results without sorting all 2.85M scores.

```cpp
for (int i = 0; i < N; i++) {
    float sim = tanimoto(query.fp, db.molecules[i].fp);
    // update min-heap of size K
}
```

**Result:**
```
Sequential : 117.45 ms  (1.00x baseline)
```
![sequential search](<image 1.png>)
---

## Step 3 — OpenMP Parallel Search

Splits the 2.85M molecule loop across all CPU cores simultaneously using `#pragma omp parallel for`.

**The Race Condition Problem:**

Multiple threads updating a shared heap simultaneously causes data corruption:
```
Thread 1: sim=0.82 → checks heap → decides to push...
Thread 2: sim=0.91 → checks heap → decides to push...
Thread 1 & 2 push simultaneously → heap corrupted ✗
```

**Solution — Local Heaps:**

Each thread maintains its own private heap, then one thread merges all local heaps at the end:
```
Thread 1 → chunk 1 → local top-K heap
Thread 2 → chunk 2 → local top-K heap
...
Thread N → chunk N → local top-K heap
                   ↓ merge
              Final top-K heap ✓
```

**Result:**

![OpenMP Results](<image 2.png>)

```
Sequential : 167.95 ms  (1.00x)
OpenMP     :  51.51 ms  (3.26x)
```

**Why 3.26x and not 8x?** Amdahl's Law — memory bandwidth is the bottleneck. All threads compete to read the same 696 MB from RAM simultaneously. WSL also adds thread management overhead vs bare Linux.

---

## Step 4 — SIMD/AVX2 Optimization

OpenMP parallelizes **across** molecules. SIMD makes **each individual Tanimoto calculation** faster.

**Normal CPU register (64-bit):** holds 1 × `uint64`

**AVX2 register (256-bit):** holds 4 × `uint64` simultaneously

```
Normal : AND(word0), AND(word1), AND(word2), AND(word3) → 4 instructions
AVX2   : AND(word0, word1, word2, word3)                → 1 instruction
```

The tanimoto loop runs in **8 AVX2 iterations** instead of 32 scalar iterations:

```cpp
for (int i = 0; i < FP_WORDS; i += 4) {
    __m256i va   = _mm256_loadu_si256((__m256i*)(&a[i]));
    __m256i vb   = _mm256_loadu_si256((__m256i*)(&b[i]));
    __m256i vand = _mm256_and_si256(va, vb);   // 4 ANDs at once
    __m256i vor  = _mm256_or_si256(va, vb);    // 4 ORs  at once
    // extract + popcount each of 4 words
}
```

**Result:**

![SIMD + OpenMP Results](<image 3.png>)

```
Sequential    : 117.45 ms  (1.00x)
OpenMP        :  61.26 ms  (1.92x)
SIMD + OpenMP :  58.42 ms  (2.01x)
```

**Why only 2.01x?** The workload is **memory-bound**, not compute-bound. The CPU already receives data from RAM faster than it can process it — speeding up computation doesn't help much when the bottleneck is memory bandwidth. The `-O2` compiler flag also auto-vectorizes the sequential loop, so explicit AVX2 adds less marginal gain.

---

## Step 5 — KMeans Clustering

Groups 2.85M molecules into **K=100 chemical clusters** using Tanimoto-based KMeans with majority-vote centroid updates.

**Algorithm:**
```
1. Pick 100 random molecules as initial centroids
2. For each iteration (max 20):
   a. Assign every molecule to its nearest centroid (Tanimoto)
   b. Recompute each centroid via majority vote on each bit
   c. If no molecule changed cluster → converged, stop early
3. Output cluster assignments
```

**Convergence** means molecules stopped switching clusters between iterations — the algorithm found a stable solution. Both sequential and parallel versions converged at **iteration 9**, confirming correctness.

**Per-iteration cost:** 2,854,639 molecules × 100 centroids = **285 million Tanimoto calls per iteration**

**Result:**

![KMeans Timing](image.png)

![KMeans Benchmark](kmeans_benchmark.png)

```
Sequential   : 191,694 ms  (1.00x)   ~3.2 minutes
OpenMP+SIMD  :  28,068 ms  (6.83x)   ~28 seconds
```

---

## Cluster Visualization

50,000 molecules sampled evenly, fingerprints reduced from **2048 dimensions → 2D** using PCA, colored by cluster assignment:

![KMeans Clusters](kmeans_clusters.png)

---

## Build & Run

```bash
# Setup
cd hpc_search
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run similarity search benchmark
./hpc_search

# Run KMeans clustering benchmark
./hpc_kmeans
```

**Requirements:** GCC 13+, CMake 3.16+, CPU with AVX2 support

---

## Full Benchmark Summary

### Similarity Search (K=5, 2.85M molecules)

| Method | Time | Speedup |
|---|---|---|
| Sequential | 117.45 ms | 1.00x |
| OpenMP | 61.26 ms | 1.92x |
| SIMD + OpenMP | 58.42 ms | 2.01x |

### KMeans Clustering (K=100, 9 iterations, 2.85M molecules)

| Method | Time | Speedup |
|---|---|---|
| Sequential | 191,694 ms | 1.00x |
| OpenMP + SIMD | 28,068 ms | 6.83x |

---

## Future Enhancements

- **CUDA GPU kernel** — offload all 2.85M Tanimoto calls to GPU simultaneously (expected 20-50x speedup)
- **Approximate Nearest Neighbor (ANN)** with Locality Sensitive Hashing for sub-linear search time
- **Butina clustering** — the standard cheminformatics clustering algorithm

---

## Technologies

`C++17` `OpenMP` `AVX2/SIMD` `Python` `RDKit` `NumPy` `scikit-learn` `CMake` `Kaggle`
