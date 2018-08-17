#pragma once
#include <iostream>
#include <ctime>
#include <cstdio>

// Benchmarking function
template <typename T, typename... O>
void timeit(T&& lambda, O&& ... params) {
    clock_t start = clock();
    std::forward<decltype(lambda)>(lambda)(std::forward<decltype(params)>(params)...);
    clock_t stop = clock();
    double elapsed = (double) (stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("Time elapsed in ms: %f\n\n", elapsed);
}