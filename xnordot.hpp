#pragma once

#define XTENSOR_USE_XSIMD
#include <iostream>
#include <vector>
#include <cmath>

// xtensor bitset simd instructions
#include <xtl/xdynamic_bitset.hpp>
#include <xsimd/memory/xsimd_aligned_allocator.hpp>

// intrinsics
#include <immintrin.h>

enum class input_alignment : bool
{
    unsafe = false,
    safe = true,
};

static constexpr int NUM_BITS = 8;
// 32 bytes because AVX256
static constexpr int ALIGN_SIZE = 32;

#define ALLIGN_ASSERT(expr) ALLIGN_ASSERT_IMPL(expr, __FILE__, __LINE__)
#define ALLIGN_ASSERT_IMPL(expr, file, line)                                                                           \
    if ((intptr_t)(expr) % ALIGN_SIZE)                                                                                 \
    {                                                                                                                  \
        throw std::runtime_error(std::string(file) + ':' + std::to_string(line) + ": Not alligned (" #expr ") \n\t");  \
    }

#define C_LAYOUT_ASSERT(expr) C_LAYOUT_ASSERT_IMPL(expr, __FILE__, __LINE__)
#define C_LAYOUT_ASSERT_IMPL(expr, file,line)                                                                          \
    if (expr.layout() != xt::layout_type::row_major)                                                                   \
    {                                                                                                                  \
        throw std::runtime_error(std::string(file) + ':' + std::to_string(line) + ": Layout not C-style (" #expr ") \n\t"); \
    }

static constexpr std::uint8_t lookup8bit[256] = {
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
        /* 10 */ 1, /* 11 */ 2, /* 12 */ 2, /* 13 */ 3,
        /* 14 */ 2, /* 15 */ 3, /* 16 */ 3, /* 17 */ 4,
        /* 18 */ 2, /* 19 */ 3, /* 1a */ 3, /* 1b */ 4,
        /* 1c */ 3, /* 1d */ 4, /* 1e */ 4, /* 1f */ 5,
        /* 20 */ 1, /* 21 */ 2, /* 22 */ 2, /* 23 */ 3,
        /* 24 */ 2, /* 25 */ 3, /* 26 */ 3, /* 27 */ 4,
        /* 28 */ 2, /* 29 */ 3, /* 2a */ 3, /* 2b */ 4,
        /* 2c */ 3, /* 2d */ 4, /* 2e */ 4, /* 2f */ 5,
        /* 30 */ 2, /* 31 */ 3, /* 32 */ 3, /* 33 */ 4,
        /* 34 */ 3, /* 35 */ 4, /* 36 */ 4, /* 37 */ 5,
        /* 38 */ 3, /* 39 */ 4, /* 3a */ 4, /* 3b */ 5,
        /* 3c */ 4, /* 3d */ 5, /* 3e */ 5, /* 3f */ 6,
        /* 40 */ 1, /* 41 */ 2, /* 42 */ 2, /* 43 */ 3,
        /* 44 */ 2, /* 45 */ 3, /* 46 */ 3, /* 47 */ 4,
        /* 48 */ 2, /* 49 */ 3, /* 4a */ 3, /* 4b */ 4,
        /* 4c */ 3, /* 4d */ 4, /* 4e */ 4, /* 4f */ 5,
        /* 50 */ 2, /* 51 */ 3, /* 52 */ 3, /* 53 */ 4,
        /* 54 */ 3, /* 55 */ 4, /* 56 */ 4, /* 57 */ 5,
        /* 58 */ 3, /* 59 */ 4, /* 5a */ 4, /* 5b */ 5,
        /* 5c */ 4, /* 5d */ 5, /* 5e */ 5, /* 5f */ 6,
        /* 60 */ 2, /* 61 */ 3, /* 62 */ 3, /* 63 */ 4,
        /* 64 */ 3, /* 65 */ 4, /* 66 */ 4, /* 67 */ 5,
        /* 68 */ 3, /* 69 */ 4, /* 6a */ 4, /* 6b */ 5,
        /* 6c */ 4, /* 6d */ 5, /* 6e */ 5, /* 6f */ 6,
        /* 70 */ 3, /* 71 */ 4, /* 72 */ 4, /* 73 */ 5,
        /* 74 */ 4, /* 75 */ 5, /* 76 */ 5, /* 77 */ 6,
        /* 78 */ 4, /* 79 */ 5, /* 7a */ 5, /* 7b */ 6,
        /* 7c */ 5, /* 7d */ 6, /* 7e */ 6, /* 7f */ 7,
        /* 80 */ 1, /* 81 */ 2, /* 82 */ 2, /* 83 */ 3,
        /* 84 */ 2, /* 85 */ 3, /* 86 */ 3, /* 87 */ 4,
        /* 88 */ 2, /* 89 */ 3, /* 8a */ 3, /* 8b */ 4,
        /* 8c */ 3, /* 8d */ 4, /* 8e */ 4, /* 8f */ 5,
        /* 90 */ 2, /* 91 */ 3, /* 92 */ 3, /* 93 */ 4,
        /* 94 */ 3, /* 95 */ 4, /* 96 */ 4, /* 97 */ 5,
        /* 98 */ 3, /* 99 */ 4, /* 9a */ 4, /* 9b */ 5,
        /* 9c */ 4, /* 9d */ 5, /* 9e */ 5, /* 9f */ 6,
        /* a0 */ 2, /* a1 */ 3, /* a2 */ 3, /* a3 */ 4,
        /* a4 */ 3, /* a5 */ 4, /* a6 */ 4, /* a7 */ 5,
        /* a8 */ 3, /* a9 */ 4, /* aa */ 4, /* ab */ 5,
        /* ac */ 4, /* ad */ 5, /* ae */ 5, /* af */ 6,
        /* b0 */ 3, /* b1 */ 4, /* b2 */ 4, /* b3 */ 5,
        /* b4 */ 4, /* b5 */ 5, /* b6 */ 5, /* b7 */ 6,
        /* b8 */ 4, /* b9 */ 5, /* ba */ 5, /* bb */ 6,
        /* bc */ 5, /* bd */ 6, /* be */ 6, /* bf */ 7,
        /* c0 */ 2, /* c1 */ 3, /* c2 */ 3, /* c3 */ 4,
        /* c4 */ 3, /* c5 */ 4, /* c6 */ 4, /* c7 */ 5,
        /* c8 */ 3, /* c9 */ 4, /* ca */ 4, /* cb */ 5,
        /* cc */ 4, /* cd */ 5, /* ce */ 5, /* cf */ 6,
        /* d0 */ 3, /* d1 */ 4, /* d2 */ 4, /* d3 */ 5,
        /* d4 */ 4, /* d5 */ 5, /* d6 */ 5, /* d7 */ 6,
        /* d8 */ 4, /* d9 */ 5, /* da */ 5, /* db */ 6,
        /* dc */ 5, /* dd */ 6, /* de */ 6, /* df */ 7,
        /* e0 */ 3, /* e1 */ 4, /* e2 */ 4, /* e3 */ 5,
        /* e4 */ 4, /* e5 */ 5, /* e6 */ 5, /* e7 */ 6,
        /* e8 */ 4, /* e9 */ 5, /* ea */ 5, /* eb */ 6,
        /* ec */ 5, /* ed */ 6, /* ee */ 6, /* ef */ 7,
        /* f0 */ 4, /* f1 */ 5, /* f2 */ 5, /* f3 */ 6,
        /* f4 */ 5, /* f5 */ 6, /* f6 */ 6, /* f7 */ 7,
        /* f8 */ 5, /* f9 */ 6, /* fa */ 6, /* fb */ 7,
        /* fc */ 6, /* fd */ 7, /* fe */ 7, /* ff */ 8
};

// Used to take the first few bits of a uint8_t
static constexpr unsigned long mask[] = {
        0x00000000UL,
        0x00000001UL,
        0x00000003UL,
        0x00000007UL,
        0x0000000fUL,
        0x0000001fUL,
        0x0000003fUL,
        0x0000007fUL,
        0x000000ffUL,
        0x000001ffUL,
        0x000003ffUL,
        0x000007ffUL,
        0x00000fffUL,
        0x00001fffUL,
        0x00003fffUL,
        0x00007fffUL,
        0x0000ffffUL,
        0x0001ffffUL,
        0x0003ffffUL,
        0x0007ffffUL,
        0x000fffffUL,
        0x001fffffUL,
        0x003fffffUL,
        0x007fffffUL,
        0x00ffffffUL,
        0x01ffffffUL,
        0x03ffffffUL,
        0x07ffffffUL,
        0x0fffffffUL,
        0x1fffffffUL,
        0x3fffffffUL,
        0x7fffffffUL,
        0xffffffffUL,
};

// pt = pack table
static constexpr unsigned pt[] = {
        0,
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
        112,
        120,
        128,
        136,
        144,
        152,
        160,
        168,
        176,
        184,
        192,
        200,
        208,
        216,
        224,
        232,
        240,
        248,
};

/// performs unsafe load sign on 8 floats at a single time and writes it into int8
/// \param data - floating point array to extract sign from
/// \param res - resulting packed bit array
/// \param size - size of data
inline void unsafe_sign(const float* data, std::uint8_t* res, std::size_t size){
    // 8 floats at a time
    // FLOAT_PACK is related to uint8_t. Change FLOAT_PACK to 16 for AVX512
    // implies we need ptrs of std::uint16_t.
    static const auto FLOAT_PACK = 8;
    const std::uint64_t limit = size - size % (FLOAT_PACK * ALIGN_SIZE);
    auto i = 0;
    for(; i < limit; i+= (FLOAT_PACK * ALIGN_SIZE)) {
        int _accum[8 + ALIGN_SIZE]= {0};
        int* accum = (int *) ((intptr_t) (_accum) + ALIGN_SIZE - (intptr_t) (_accum) % ALIGN_SIZE);

        ALLIGN_ASSERT(accum)
        // Load all 32 * 32 floats.
        __m256 res1 = _mm256_loadu_ps(data + i + pt[0]);
        __m256 res2 = _mm256_loadu_ps(data + i + pt[1]);
        __m256 res3 = _mm256_loadu_ps(data + i + pt[2]);
        __m256 res4 = _mm256_loadu_ps(data + i + pt[3]);
        __m256 res5 = _mm256_loadu_ps(data + i + pt[4]);
        __m256 res6 = _mm256_loadu_ps(data + i + pt[5]);
        __m256 res7 = _mm256_loadu_ps(data + i + pt[6]);
        __m256 res8 = _mm256_loadu_ps(data + i + pt[7]);
        __m256 res9 = _mm256_loadu_ps(data + i + pt[8]);
        __m256 res10 = _mm256_loadu_ps(data + i + pt[9]);
        __m256 res11 = _mm256_loadu_ps(data + i + pt[10]);
        __m256 res12 = _mm256_loadu_ps(data + i + pt[11]);
        __m256 res13 = _mm256_loadu_ps(data + i + pt[12]);
        __m256 res14 = _mm256_loadu_ps(data + i + pt[13]);
        __m256 res15 = _mm256_loadu_ps(data + i + pt[14]);
        __m256 res16 = _mm256_loadu_ps(data + i + pt[15]);

        accum[0] |= _mm256_movemask_ps(res1) << pt[3];
        accum[0] |= _mm256_movemask_ps(res2) << pt[2];
        accum[0] |= _mm256_movemask_ps(res3) << pt[1];
        accum[0] |= _mm256_movemask_ps(res4) << pt[0];

        accum[1] |= _mm256_movemask_ps(res5) << pt[3];
        accum[1] |= _mm256_movemask_ps(res6) << pt[2];
        accum[1] |= _mm256_movemask_ps(res7) << pt[1];
        accum[1] |= _mm256_movemask_ps(res8) << pt[0];

        accum[2] |= _mm256_movemask_ps(res9) << pt[3];
        accum[2] |= _mm256_movemask_ps(res10) << pt[2];
        accum[2] |= _mm256_movemask_ps(res11) << pt[1];
        accum[2] |= _mm256_movemask_ps(res12) << pt[0];

        accum[3] |= _mm256_movemask_ps(res13) << pt[3];
        accum[3] |= _mm256_movemask_ps(res14) << pt[2];
        accum[3] |= _mm256_movemask_ps(res15) << pt[1];
        accum[3] |= _mm256_movemask_ps(res16) << pt[0];

        __m256 res17 = _mm256_loadu_ps(data + i + pt[16]);
        __m256 res18 = _mm256_loadu_ps(data + i + pt[17]);
        __m256 res19 = _mm256_loadu_ps(data + i + pt[18]);
        __m256 res20 = _mm256_loadu_ps(data + i + pt[19]);
        __m256 res21 = _mm256_loadu_ps(data + i + pt[20]);
        __m256 res22 = _mm256_loadu_ps(data + i + pt[21]);
        __m256 res23 = _mm256_loadu_ps(data + i + pt[22]);
        __m256 res24 = _mm256_loadu_ps(data + i + pt[23]);
        __m256 res25 = _mm256_loadu_ps(data + i + pt[24]);
        __m256 res26 = _mm256_loadu_ps(data + i + pt[25]);
        __m256 res27 = _mm256_loadu_ps(data + i + pt[26]);
        __m256 res28 = _mm256_loadu_ps(data + i + pt[27]);
        __m256 res29 = _mm256_loadu_ps(data + i + pt[28]);
        __m256 res30 = _mm256_loadu_ps(data + i + pt[29]);
        __m256 res31 = _mm256_loadu_ps(data + i + pt[30]);
        __m256 res32 = _mm256_loadu_ps(data + i + pt[31]);

        accum[4] |= _mm256_movemask_ps(res17) << pt[3];
        accum[4] |= _mm256_movemask_ps(res18) << pt[2];
        accum[4] |= _mm256_movemask_ps(res19) << pt[1];
        accum[4] |= _mm256_movemask_ps(res20) << pt[0];

        accum[5] |= _mm256_movemask_ps(res21) << pt[3];
        accum[5] |= _mm256_movemask_ps(res22) << pt[2];
        accum[5] |= _mm256_movemask_ps(res23) << pt[1];
        accum[5] |= _mm256_movemask_ps(res24) << pt[0];

        accum[6] |= _mm256_movemask_ps(res25) << pt[3];
        accum[6] |= _mm256_movemask_ps(res26) << pt[2];
        accum[6] |= _mm256_movemask_ps(res27) << pt[1];
        accum[6] |= _mm256_movemask_ps(res28) << pt[0];

        accum[7] |= _mm256_movemask_ps(res29) << pt[3];
        accum[7] |= _mm256_movemask_ps(res30) << pt[2];
        accum[7] |= _mm256_movemask_ps(res31) << pt[1];
        accum[7] |= _mm256_movemask_ps(res32) << pt[0];

        __m256i chunk = _mm256_load_si256((__m256i *) accum);
        _mm256_store_si256((__m256i*) (res + i/FLOAT_PACK), chunk);
    }

    for(; i < size - size % FLOAT_PACK; i+=FLOAT_PACK) {
        __m256 accum = _mm256_loadu_ps(data + i);
        res[i/FLOAT_PACK] = (std::uint8_t) _mm256_movemask_ps(accum);
    }

    // If there are any remainders bit-wise
    if(size - i) {
        std::uint8_t residue = 0;
        auto base = i;
        for (; i < size; i++) {
            residue |= std::signbit(data[i]) << i - base;
        }
        res[i / FLOAT_PACK] = residue;
    }
}

/// performs sign on 8 floats at a single time and writes it into int8
/// \param data - floating point array to extract sign from
/// \param res - resulting packed bit array
/// \param size - size of data
inline void sign(const float* data, std::uint8_t* res, std::size_t size){
    ALLIGN_ASSERT(data)
    ALLIGN_ASSERT(res)
    // 8 floats at a time
    // FLOAT_PACK is related to uint8_t. Change FLOAT_PACK to 16 for AVX512
    // implies we need ptrs of std::uint16_t.
    static const auto FLOAT_PACK = 8;
    const std::uint64_t limit = size - size % (FLOAT_PACK * 32);
    auto i = 0;
    for(; i < limit; i+= (FLOAT_PACK * 32)) {
        // In order to keep allignment, we must over-allocate by ALIGN_SIZE
        int _accum[8 + ALIGN_SIZE]= {0};
        int* accum = (int*) ((intptr_t)(_accum) + ALIGN_SIZE - (intptr_t)(_accum) % ALIGN_SIZE);
        ALLIGN_ASSERT(accum)
        // Load all 32 * 32 floats.
        __m256 res1 = _mm256_load_ps(data + i + pt[0]);
        __m256 res2 = _mm256_load_ps(data + i + pt[1]);
        __m256 res3 = _mm256_load_ps(data + i + pt[2]);
        __m256 res4 = _mm256_load_ps(data + i + pt[3]);
        __m256 res5 = _mm256_load_ps(data + i + pt[4]);
        __m256 res6 = _mm256_load_ps(data + i + pt[5]);
        __m256 res7 = _mm256_load_ps(data + i + pt[6]);
        __m256 res8 = _mm256_load_ps(data + i + pt[7]);
        __m256 res9 = _mm256_load_ps(data + i + pt[8]);
        __m256 res10 = _mm256_load_ps(data + i + pt[9]);
        __m256 res11 = _mm256_load_ps(data + i + pt[10]);
        __m256 res12 = _mm256_load_ps(data + i + pt[11]);
        __m256 res13 = _mm256_load_ps(data + i + pt[12]);
        __m256 res14 = _mm256_load_ps(data + i + pt[13]);
        __m256 res15 = _mm256_load_ps(data + i + pt[14]);
        __m256 res16 = _mm256_load_ps(data + i + pt[15]);

        accum[0] |= _mm256_movemask_ps(res1) << pt[3];
        accum[0] |= _mm256_movemask_ps(res2) << pt[2];
        accum[0] |= _mm256_movemask_ps(res3) << pt[1];
        accum[0] |= _mm256_movemask_ps(res4) << pt[0];

        accum[1] |= _mm256_movemask_ps(res5) << pt[3];
        accum[1] |= _mm256_movemask_ps(res6) << pt[2];
        accum[1] |= _mm256_movemask_ps(res7) << pt[1];
        accum[1] |= _mm256_movemask_ps(res8) << pt[0];

        accum[2] |= _mm256_movemask_ps(res9) << pt[3];
        accum[2] |= _mm256_movemask_ps(res10) << pt[2];
        accum[2] |= _mm256_movemask_ps(res11) << pt[1];
        accum[2] |= _mm256_movemask_ps(res12) << pt[0];

        accum[3] |= _mm256_movemask_ps(res13) << pt[3];
        accum[3] |= _mm256_movemask_ps(res14) << pt[2];
        accum[3] |= _mm256_movemask_ps(res15) << pt[1];
        accum[3] |= _mm256_movemask_ps(res16) << pt[0];

        __m256 res17 = _mm256_load_ps(data + i + pt[16]);
        __m256 res18 = _mm256_load_ps(data + i + pt[17]);
        __m256 res19 = _mm256_load_ps(data + i + pt[18]);
        __m256 res20 = _mm256_load_ps(data + i + pt[19]);
        __m256 res21 = _mm256_load_ps(data + i + pt[20]);
        __m256 res22 = _mm256_load_ps(data + i + pt[21]);
        __m256 res23 = _mm256_load_ps(data + i + pt[22]);
        __m256 res24 = _mm256_load_ps(data + i + pt[23]);
        __m256 res25 = _mm256_load_ps(data + i + pt[24]);
        __m256 res26 = _mm256_load_ps(data + i + pt[25]);
        __m256 res27 = _mm256_load_ps(data + i + pt[26]);
        __m256 res28 = _mm256_load_ps(data + i + pt[27]);
        __m256 res29 = _mm256_load_ps(data + i + pt[28]);
        __m256 res30 = _mm256_load_ps(data + i + pt[29]);
        __m256 res31 = _mm256_load_ps(data + i + pt[30]);
        __m256 res32 = _mm256_load_ps(data + i + pt[31]);

        accum[4] |= _mm256_movemask_ps(res17) << pt[3];
        accum[4] |= _mm256_movemask_ps(res18) << pt[2];
        accum[4] |= _mm256_movemask_ps(res19) << pt[1];
        accum[4] |= _mm256_movemask_ps(res20) << pt[0];

        accum[5] |= _mm256_movemask_ps(res21) << pt[3];
        accum[5] |= _mm256_movemask_ps(res22) << pt[2];
        accum[5] |= _mm256_movemask_ps(res23) << pt[1];
        accum[5] |= _mm256_movemask_ps(res24) << pt[0];

        accum[6] |= _mm256_movemask_ps(res25) << pt[3];
        accum[6] |= _mm256_movemask_ps(res26) << pt[2];
        accum[6] |= _mm256_movemask_ps(res27) << pt[1];
        accum[6] |= _mm256_movemask_ps(res28) << pt[0];

        accum[7] |= _mm256_movemask_ps(res29) << pt[3];
        accum[7] |= _mm256_movemask_ps(res30) << pt[2];
        accum[7] |= _mm256_movemask_ps(res31) << pt[1];
        accum[7] |= _mm256_movemask_ps(res32) << pt[0];

        __m256i chunk = _mm256_load_si256((__m256i *) accum);
        _mm256_store_si256((__m256i*) (res + i/FLOAT_PACK), chunk);
    }

    for(; i < size - size % FLOAT_PACK; i+=FLOAT_PACK) {
        __m256 accum = _mm256_load_ps(data + i);
        res[i/FLOAT_PACK] = (std::uint8_t) _mm256_movemask_ps(accum);
    }

    // If there are any remainders bit-wise
    if(size - i) {
        std::uint8_t residue = 0;
        auto base = i;
        for (; i < size; i++) {
            residue |= std::signbit(data[i]) << i - base;
        }
        res[i / FLOAT_PACK] = residue;
    }
}

/// Performs 32 byte xors at a single time.
/// \param x - left operand data
/// \param y - right operand data
/// \param res - result buffer to save into
/// \param size - size of x and y. IMPORTANT: IN BITS.
inline void xnor(std::uint8_t* x, std::uint8_t* y, std::uint8_t* res, std::size_t size){
    ALLIGN_ASSERT(x)
    ALLIGN_ASSERT(y)
    ALLIGN_ASSERT(res)
    static const auto UINT8_PACK = 32; // Can fit 32 uint8_t's
    // 8 elements per byte. Adding by uint8_t will add 8 at a time.
    static const auto SIZE_SCALE = 8;
    const std::uint64_t limit = size - size % (UINT8_PACK * SIZE_SCALE); // 32 uint8_t's at a time
    auto i = 0;
    for(; i < limit; i+=UINT8_PACK*SIZE_SCALE) {
        __m256i tmp_x = _mm256_load_si256((__m256i *) (x + i/SIZE_SCALE));
        __m256i tmp_y = _mm256_load_si256((__m256i *) (y + i/SIZE_SCALE));
        _mm256_store_si256((__m256i *) (res + i/SIZE_SCALE), ~_mm256_xor_si256(tmp_x, tmp_y));
    }
    // The remaining <32 uint8_t's are computed in sequence
    auto residue = size % SIZE_SCALE;
    for(; i < size - residue; i+=SIZE_SCALE){
        res[i/SIZE_SCALE] = ~(x[i/SIZE_SCALE] ^ y[i/SIZE_SCALE]);
    }
    // The remainder bits are computed with mask
    if (residue != 0)
        res[i / SIZE_SCALE] = ~(x[i / SIZE_SCALE] ^ y[i / SIZE_SCALE]) & mask[residue];
}


// Implementation of harley-seal
namespace popcnt{
    __m256i popcount(const __m256i v)
    {
        const __m256i m1 = _mm256_set1_epi8(0x55);
        const __m256i m2 = _mm256_set1_epi8(0x33);
        const __m256i m4 = _mm256_set1_epi8(0x0F);

        const __m256i t1 = _mm256_sub_epi8(v,       (_mm256_srli_epi16(v,  1) & m1));
        const __m256i t2 = _mm256_add_epi8(t1 & m2, (_mm256_srli_epi16(t1, 2) & m2));
        const __m256i t3 = _mm256_add_epi8(t2, _mm256_srli_epi16(t2, 4)) & m4;
        return _mm256_sad_epu8(t3, _mm256_setzero_si256());
    }

    void CSA(__m256i& h, __m256i& l, __m256i a, __m256i b, __m256i c)
    {
        const __m256i u = a ^ b;
        h = (a & b) | (u & c);
        l = u ^ c;
    }

    std::uint64_t popcnt(const __m256i* data, const std::uint64_t size)
    {
        __m256i total     = _mm256_setzero_si256();
        __m256i ones      = _mm256_setzero_si256();
        __m256i twos      = _mm256_setzero_si256();
        __m256i fours     = _mm256_setzero_si256();
        __m256i eights    = _mm256_setzero_si256();
        __m256i sixteens  = _mm256_setzero_si256();
        __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

        const std::uint64_t limit = size - size % 16;
        auto i = 0;
        for(; i < limit; i += 16)
        {
            CSA(twosA, ones, ones, data[i+0], data[i+1]);
            CSA(twosB, ones, ones, data[i+2], data[i+3]);
            CSA(foursA, twos, twos, twosA, twosB);
            CSA(twosA, ones, ones, data[i+4], data[i+5]);
            CSA(twosB, ones, ones, data[i+6], data[i+7]);
            CSA(foursB, twos, twos, twosA, twosB);
            CSA(eightsA,fours, fours, foursA, foursB);
            CSA(twosA, ones, ones, data[i+8], data[i+9]);
            CSA(twosB, ones, ones, data[i+10], data[i+11]);
            CSA(foursA, twos, twos, twosA, twosB);
            CSA(twosA, ones, ones, data[i+12], data[i+13]);
            CSA(twosB, ones, ones, data[i+14], data[i+15]);
            CSA(foursB, twos, twos, twosA, twosB);
            CSA(eightsB, fours, fours, foursA, foursB);
            CSA(sixteens, eights, eights, eightsA, eightsB);

            total = _mm256_add_epi64(total, popcount(sixteens));
        }

        if (limit != 0) {
            total = _mm256_slli_epi64(total, 4);     // * 16
            total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(eights), 3)); // += 8 * ...
            total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(fours), 2)); // += 4 * ...
            total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(twos), 1)); // += 2 * ...
            total = _mm256_add_epi64(total, popcount(ones));
        }

        for(; i < size; i++) {
            auto res = popcount(data[i]);
            total = _mm256_add_epi64(total, res);
        }


        return static_cast<std::uint64_t>(_mm256_extract_epi64(total, 0))
               + static_cast<std::uint64_t>(_mm256_extract_epi64(total, 1))
               + static_cast<std::uint64_t>(_mm256_extract_epi64(total, 2))
               + static_cast<std::uint64_t>(_mm256_extract_epi64(total, 3));
    }
} // popcnt

/// Performs popcount (population count) on bits
/// \param data - bits of data
/// \param size - length of data. IMPORTANT: IN BITS.
inline long long sum(const std::uint8_t* data, const std::size_t size)
{
    ALLIGN_ASSERT(data)
    static constexpr auto UINT8_PACK = 32;
    static constexpr auto SIZE_SCALE = 8;
    // The block remainder is taken care of in popcnt
    auto byte_size = size / SIZE_SCALE;
    auto mm256_block = byte_size / UINT8_PACK;
    auto total = popcnt::popcnt((const __m256i*) data, mm256_block);

    auto residue = mm256_block*UINT8_PACK;
    auto i = residue;
    for (; i < residue + (size - residue*SIZE_SCALE)/SIZE_SCALE; i++)
        total += lookup8bit[data[i]];

    // Prevent modulo by 0 by adding 1 to residue
    if(size - i*SIZE_SCALE) {
        auto bit_residue = size - i * SIZE_SCALE;
        if (bit_residue)
            total += lookup8bit[data[i] & mask[bit_residue]];
    }

    // size - total = # zeros (-1's)
    // total = # ones (1's)
    // sum = #1's * 1 - #-1's * 1
    return 2 * total - size;
}

/// Performs an xnordot on the given xt::xarrays
/// \param a1 - xarray to compute dot product
/// \param a2 - xarray to compute dot product
inline long long xnordot(const xt::xarray<float>& a1,
const xt::xarray<float>& a2,
::input_alignment alignment = ::input_alignment::safe){
    // Enforce alignment
    xtl::xdynamic_bitset<std::uint8_t, xsimd::aligned_allocator<std::uint8_t, 32>> bitset_a1;
    xtl::xdynamic_bitset<std::uint8_t, xsimd::aligned_allocator<std::uint8_t, 32>> bitset_a2;
    xtl::xdynamic_bitset<std::uint8_t, xsimd::aligned_allocator<std::uint8_t, 32>> res;
    const auto RESULT_SIZE = a1.size();
    bitset_a1.reserve(RESULT_SIZE);
    bitset_a2.reserve(RESULT_SIZE);
    res.reserve(RESULT_SIZE);

    if(alignment == ::input_alignment::safe) {
        ALLIGN_ASSERT(a1.data())
        ALLIGN_ASSERT(a2.data())
        sign(a1.data(), bitset_a1.data(), RESULT_SIZE);
        sign(a2.data(), bitset_a2.data(), RESULT_SIZE);
    }
    else{
        unsafe_sign(a1.data(), bitset_a1.data(), RESULT_SIZE);
        unsafe_sign(a2.data(), bitset_a2.data(), RESULT_SIZE);
    }
    xnor(bitset_a1.data(), bitset_a2.data(), res.data(), RESULT_SIZE);
    return sum(res.data(), RESULT_SIZE);
}
