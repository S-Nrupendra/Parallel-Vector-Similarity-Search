This project uses concepts of High performance computing (HPC) like OpenMp to find similar vectors in a dataset which has millions of vectors
This has directly use in LLMs & RAGs

For this project we are using Chemistry Dataset instead of official datasets like SIFT1M ( http://corpus-texmex.irisa.fr/)

Local (WSL)  → Step 1: I/O & Architecture
             → Step 2: Sequential baseline  
             → Step 3: OpenMP
             → Step 4: SIMD/AVX

Kaggle GPU   → Step 5: CUDA (later)

Step 0 : Data Preprocessing
* We are using a Chemistry Dataset (https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)
* Using the Python library `rdkit` we are able to obtain fingerprints for different molecules and `.bin` file

Step 1 : I/O Architecture
1) We need to load the data into memory(the .bin file)
2) Define the data structures everyone will use
3) Write a slow but correct sequential search as baseline
4) Measure how long it takes -> this becomes our "1x" reference
    Order:
        1) `include/fingerprint_db.h` — Shared Data Structures
        2) `src/io.cpp` - Loading the Binary File
        3) `src/search_sequential.cpp` — Brute Force Search
        4) `src/main.cpp` - Entry Point & Timer
        5) `CMakeLists.txt` - Build Instructions
            * This tells cmake how to compile everything
            * -O2 → compiler optimizations on
            * -mavx2 → enable AVX2 instructions (needed for SIMD later)
            * -fopenmp → enable OpenMP (needed for parallel later)
            * Compile the 3 files together into one program called `hpc_search`

`
# Go back to the right place
cd "/mnt/c/Users/snrup/OneDrive/Desktop/Sem 6/HPC Project/hpc_search"

# Create build folder inside hpc_search
mkdir -p build && cd build

# Now cmake can find src/ correctly
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
./hpc_search
`

Step 2: OpenMp
With sequential we are computing Tanimoto One molecule at a time, on one CPU Core, where as all the other cores sitting idle while this happens

Using OpenMp we will utilize all the cores available in CPU.

If a CPU has 8 Cores :
`
Core 1 → molecules 0       to 356829
Core 2 → molecules 356830  to 713659
Core 3 → molecules 713660  to 1070489
Core 4 → molecules 1070490 to 1427319
Core 5 → molecules 1427320 to 1784149
Core 6 → molecules 1784150 to 2140979
Core 7 → molecules 2140980 to 2497809
Core 8 → molecules 2497810 to 2854638
`

By adding just one line `#pragma omp parallel for`

The Tricky Part — Race Conditions
Here's the problem. In sequential search, one thread updates the heap:
`Thread 1: sim=0.82 → heap is full → pop lowest → push 0.82 ✓`
With multiple threads doing this simultaneously:
`
Thread 1: sim=0.82 → checks heap → decides to push...
Thread 2: sim=0.91 → checks heap → decides to push...
Thread 1: pushes  → heap now corrupted ← RACE CONDITION
Thread 2: pushes  → wrong result
`
Two threads writing to the same heap at the same time = corrupted data = wrong results.

How We Solve It — Local Heaps
Each thread gets its own private heap:
`
Thread 1 → searches its chunk → builds local top-5
Thread 2 → searches its chunk → builds local top-5
Thread 3 → searches its chunk → builds local top-5
...
Thread 8 → searches its chunk → builds local top-5
`
Then at the end, one thread merges all 8 local top-5 heaps into one final top-5:
`8 local heaps (each size 5) → merge → 1 final heap (size 5)`
This is safe because no two threads ever touch the same heap.

Code Structure :
    1) `src/search_openmp.cpp`
    2) Updating `include/fingerprint_db.h`
    3) Updating `src/main.cpp`
    4) Updating `CMakeLists.txt`
        `
        cd "/mnt/c/Users/snrup/OneDrive/Desktop/Sem 6/HPC Project/hpc_search/build"
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j4
        ./hpc_search
        `
Output Observation :
Why 3.26x and not 8x?
You might wonder — if OpenMP uses multiple cores, why not a perfect 8x speedup? This is explained by Amdahl's Law (which you studied in HPC theory):
Reasons for less than perfect speedup:
1. WSL overhead — You're running on Windows Subsystem for Linux. WSL doesn't give full native Linux performance. Thread management is slower than bare metal Linux.
2. Memory bandwidth bottleneck — All threads are reading from the same 696MB file in RAM simultaneously. RAM has limited bandwidth — all cores compete for it, so they end up waiting on memory, not computing.
3. Merge overhead — After parallel search, one thread merges all local heaps. That part is sequential.
4. Thread creation cost — Spawning and synchronizing threads has a fixed overhead every time you call search_openmp().
In a real Linux machine you'd likely see 5-7x. 3.26x on WSL is completely reasonable and expected.

