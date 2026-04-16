cmake_minimum_required(VERSION 3.16)
project(hpc_search)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mfma -fopenmp -mpopcnt")

include_directories(include)

# Executable 1 — Search benchmark
add_executable(hpc_search
    src/main.cpp
    src/io.cpp
    src/search_sequential.cpp
    src/search_openmp.cpp
    src/search_simd.cpp
)
target_link_libraries(hpc_search gomp)

# Executable 2 — KMeans benchmark
add_executable(hpc_kmeans
    src/kmeans_main.cpp
    src/io.cpp
    src/kmeans_sequential.cpp
    src/kmeans_openmp_simd.cpp
)
target_link_libraries(hpc_kmeans gomp)