cmake_minimum_required(VERSION 3.10)
project(bit_packing)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcblas -march=haswell -ffast-math -funroll-loops -g -O3")

add_executable(bit_packing main.cpp timeit.hpp xnordot.hpp xnorgemm.hpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtl/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor-python/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/xsimd/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor-blas/include)
