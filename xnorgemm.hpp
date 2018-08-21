#pragma once
#define XTENSOR_USE_XSIMD
#include <iostream>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

// xtensor bitset simd instructions
#include <xtl/xdynamic_bitset.hpp>
#include <xsimd/memory/xsimd_aligned_allocator.hpp>

#include "xnordot.hpp"

using bitset_t = xtl::xdynamic_bitset<std::uint8_t, xsimd::aligned_allocator<std::uint8_t, 32>>;
static constexpr unsigned long int BLOCK_SIZE = 1000;

/// Performs an xnorgemm on the given xt::xarrays
/// \param a1 - xarray to compute gemm
/// \param a2 - xarray to compute gemm
inline xt::xarray<float> xnorgemm(const xt::xarray<float>& a1,
const xt::xarray<float>& _a2
){
    xt::xarray<float> a2 = xt::transpose(_a2);
    // Check allignment
    C_LAYOUT_ASSERT(a1)
    C_LAYOUT_ASSERT(a2)
    ALLIGN_ASSERT(a1.data())
    ALLIGN_ASSERT(a2.data())
    // Check size
    XTENSOR_ASSERT(a1.dimension() == 2 && a2.dimension() == 2)

    const auto in_shape_a1 = a1.shape();
    const auto in_shape_a2 = a2.shape();
    const auto result_shape = decltype(in_shape_a1) {in_shape_a1[0], in_shape_a2[0]};

    // Enforce alignment
    std::vector<bitset_t> bitset_a1(in_shape_a1[0], bitset_t(in_shape_a1[1]));
    // Transpose it
    std::vector<bitset_t> bitset_a2(in_shape_a2[0], bitset_t(in_shape_a2[1]));
    xt::xarray<float> res;
    res.resize(result_shape);

    for(int i = 0; i < in_shape_a1[0]; i++){
        // Reserve space for appropriate # of bits
        bitset_a1[i].resize(in_shape_a1[1]);
        auto offset = i * in_shape_a1[1];
        unsafe_sign(a1.data() + offset, bitset_a1[i].data(), in_shape_a1[1]);
    }

    for(int i = 0; i < in_shape_a2[0]; i++){
        // Reserve space for appropriate # of bits
        bitset_a2[i].resize(in_shape_a2[1]);
        auto offset = i * in_shape_a2[1];
        unsafe_sign(a2.data() + offset, bitset_a2[i].data(), in_shape_a2[1]);
    }

//    for(auto& x : bitset_a1){
//        for(auto y : x){
//            std::cout << y << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << "\n\n";
//    for(auto& x : bitset_a2){
//        for(auto y : x){
//            std::cout << y << " ";
//        }
//        std::cout << std::endl;
//    }

    // WARNING: Extremely naive kernel - no blocking, no unrolling, only transpose
    auto col_size = in_shape_a1[1];

    // This subroutine takes roughly 80%-90% of the runtime
    auto rows = in_shape_a1[0];
    auto cols = in_shape_a2[0];
//    for(int ii = 0; ii < rows; ii += block_size) {
//        for (int jj = 0; jj < cols; jj += block_size) {
//            for (int i = 0; i < std::min(block_size, rows - ii); i++) {
//                for (int j = 0; j < std::min(block_size, cols - jj); j++) {
//                    int _i = i + ii;
//                    int _j = j + jj;
//                    bitset_t temp(col_size);
//                    xnor(bitset_a1[_i].data(), bitset_a2[_j].data(), temp.data(), col_size);
//                    res(_i, _j) = sum(temp.data(), col_size);
//                }
//            }
//        }
//    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            bitset_t temp(col_size);
            xnor(bitset_a1[i].data(), bitset_a2[j].data(), temp.data(), col_size);
            res(i, j) = sum(temp.data(), col_size);
        }
    }

    return res;
}
