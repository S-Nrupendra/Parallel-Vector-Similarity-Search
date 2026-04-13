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

Step 1 :
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