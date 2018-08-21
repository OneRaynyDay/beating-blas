// Define this at the top to avoid collisions
#define XTENSOR_USE_XSIMD
#include <iostream>
#include <vector>
#include <cmath>

// xtensor-blas benchmark
#include <xtensor-blas/xlinalg.hpp>

// xtensor
#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>

// Require this for all float input
#include <xsimd/memory/xsimd_aligned_allocator.hpp>

#include "xnordot.hpp"
#include "xnorgemm.hpp"

#include "timeit.hpp"

// I am running on a computer with AVX2 instructions:
// (idp3) ❯ sysctl -a | grep avx
// hw.optional.avx1_0: 1
// hw.optional.avx2_0: 1
// hw.optional.avx512f: 0
// hw.optional.avx512cd: 0
// hw.optional.avx512dq: 0
// hw.optional.avx512bw: 0
// hw.optional.avx512vl: 0
// hw.optional.avx512ifma: 0
// hw.optional.avx512vbmi: 0

void print_vector(const xt::xarray<bool>& b){
    for(bool x : b){
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

void print_bitset(std::uint8_t b){
    for(auto i = NUM_BITS-1; i >= 0; i--){
        std::cout << ((b & (1<<i)) >> i) << " ";
    }
    std::cout << std::endl;
}

// === dot benchmark functions ===

auto blas_dot(const xt::xarray<float>& a1, const xt::xarray<float>& a2){
    return xt::linalg::dot(a1, a2);
}

auto blas_sign_dot(const xt::xarray<float>& a1, const xt::xarray<float>& a2){
    auto&& i8a1 = xt::cast<std::int8_t>(xt::sign(a1));
    auto&& i8a2 = xt::cast<std::int8_t>(xt::sign(a2));
    return xt::linalg::dot(i8a1, i8a2);
}

void benchmark_dot(){
    auto ARR_SIZE = 1UL;
    for (int iter = 0; iter < 8; iter++) {
        std::cout << "====== Array size : " << ARR_SIZE << " ====== " << std::endl;
        // We produce an alternating sequence of 0's and 1's. We need to make it start at 1
        // because sometimes the compiler implicitly turns it to a positive 0 even though it should be negative.
        xt::xarray<float> arr1;
        arr1.resize({ARR_SIZE});
        int _sign1 = 1;
        for (int i = 1; i <= ARR_SIZE; i++) {
            arr1(i-1) = i * _sign1;
            _sign1 *= -1;
        }

        xt::xarray<float> arr2;
        arr2.resize({ARR_SIZE});
        // sign starts as 1 for odd, -1 for even
        // This means the bits are either exactly the same (odd)
        // or flipped (even)
        int _sign2 = 1 - 2 * (iter % 2);
        for (int i = 1; i <= ARR_SIZE; i++) {
            arr2(i-1) = i * _sign2;
            _sign2 *= -1;
        }

        // If iter is odd, then we have a bunch of 0's summed = 0
        // If iter is even, then we have a bunch of 1's summed = ARR_SIZE
        // This adds overhead - FOR UNIT TESTING ONLY
        auto run = [&](const xt::xarray<float>& x, const xt::xarray<float>& y){
            auto res = xnordot(x, y);
            if (res != 2 * (((iter + 1) % 2) * ARR_SIZE) - ARR_SIZE) {
                throw std::runtime_error("Something's wrong.");
            }
        };

        run(arr1, arr2);

        std::cout << "=== hand-tuned bitpack+xnor+popcount safe ===" << std::endl;
        timeit(xnordot, arr1, arr2, ::input_alignment::safe);

        std::cout << "=== hand-tuned bitpack+xnor+popcount unsafe ===" << std::endl;
        timeit(xnordot, arr1, arr2, ::input_alignment::unsafe);

        std::cout << "=== xtensor-blas dot product ===" << std::endl;
        timeit(blas_dot, arr1, arr2);

        std::cout << "=== xtensor-blas dot product on bools ===" << std::endl;
        timeit(blas_sign_dot, arr1, arr2);

        // Increase the magnitude after every iteration
        ARR_SIZE *= 10;
    }
}

// === gemm benchmark functions ===

auto blas_gemm(const xt::xarray<float>& a1, const xt::xarray<float>& a2){
    return xt::linalg::dot(a1, a2);
}

auto blas_dot_gemm(const xt::xarray<float>& a1, const xt::xarray<float>& a2){
    auto&& i8a1 = xt::cast<std::int8_t>(xt::sign(a1));
    auto&& i8a2 = xt::cast<std::int8_t>(xt::sign(a2));
    return xt::linalg::dot(i8a1, i8a2);
}

void benchmark_gemm(){
    auto ARR_SIZE = 1UL;
    for (int iter = 0; iter < 12; iter++) {
        std::cout << "====== Array size : " << ARR_SIZE << " ====== " << std::endl;
        // We produce an alternating sequence of 0's and 1's. We need to make it start at 1
        // because sometimes the compiler implicitly turns it to a positive 0 even though it should be negative.
        auto X_SIZE = ARR_SIZE;
        auto Y_SIZE = ARR_SIZE;
        auto Z_SIZE = ARR_SIZE;

        xt::xarray<float> arr1;
        arr1.resize({X_SIZE, Y_SIZE});
        int _sign1 = 1;
        for (int i = 1; i <= X_SIZE; i++) {
            for (int j = 0; j < Y_SIZE; j++) {
                arr1(i - 1, j) = i * _sign1;
                _sign1 *= -1;
            }
        }

        // Take the exact opposite values
        xt::xarray<float> arr2;
        arr2.resize({Y_SIZE, Z_SIZE});
        int _sign2 = 1;
        for (int i = 1; i <= Y_SIZE; i++) {
            for (int j = 0; j < Z_SIZE; j++) {
                arr2(i - 1, j) = i * _sign2;
                _sign2 *= -1;
            }
        }

        std::cout << "=== hand-tuned bitpack+xnor+popcount (unsafe) ===" << std::endl;
        timeit(xnorgemm, arr1, arr2);

        std::cout << "=== xtensor-blas gemm ===" << std::endl;
        timeit(blas_gemm, arr1, arr2);

        std::cout << "=== xtensor-blas gemm on bools ===" << std::endl;
        timeit(blas_dot_gemm, arr1, arr2);

        // Increase the magnitude after every iteration
        ARR_SIZE *= 2;
    }
}

// ===

int main() {
//    benchmark_dot();
//    benchmark_gemm();

    // Unit tests
    auto test_iters = 100;
    for(auto i = 0; i < test_iters; i++){
        xt::xarray<float, xt::layout_type::row_major> a1 = xt::random::rand<float>({100, 250}, -1000.0f, 1000.0f);
        xt::xarray<float, xt::layout_type::row_major> a2 = xt::random::rand<float>({250, 30}, -1000.0f, 1000.0f);
        auto res = xt::linalg::dot(xt::sign(a1), xt::sign(a2));
        auto xnorgemm_res = xnorgemm(a1, a2);
        assert(xt::allclose(res, xnorgemm_res));
    }
}
