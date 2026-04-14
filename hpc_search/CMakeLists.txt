cmake_minimum_required(VERSION 3.16)
project(hpc_search)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mfma -fopenmp")

include_directories(include)

add_executable(hpc_search
    src/main.cpp
    src/io.cpp
    src/search_sequential.cpp
    src/search_openmp.cpp
)

target_link_libraries(hpc_search gomp)