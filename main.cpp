#include <iostream>
#include <vector>

// xtensor
#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xdynamic_bitset.hpp>

// intrinsics
#include <immintrin.h>

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

static const int NUM_BITS = 8;

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

// performs sign on 8 floats at a single time and writes it into int8
void sign(float* data, std::uint8_t* res, std::size_t n){
    __m256 tmp = _mm256_load_ps(data);
    *res = (std::uint8_t) _mm256_movemask_ps(tmp);
}

// Performs 32 byte xors at a single time.
void xnor(std::uint8_t* x, std::uint8_t* y, std::uint8_t* res){
    __m256i tmp_x = _mm256_load_si256((__m256i*) x);
    __m256i tmp_y = _mm256_load_si256((__m256i*) y);
    *(__m256i*) res = ~_mm256_xor_si256(tmp_x, tmp_y);
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

uint64_t popcnt(const __m256i* data, const uint64_t size)
{
    __m256i total     = _mm256_setzero_si256();
    __m256i ones      = _mm256_setzero_si256();
    __m256i twos      = _mm256_setzero_si256();
    __m256i fours     = _mm256_setzero_si256();
    __m256i eights    = _mm256_setzero_si256();
    __m256i sixteens  = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    const uint64_t limit = size - size % 16;
    uint64_t i = 0;

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

    total = _mm256_slli_epi64(total, 4);     // * 16
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(eights), 3)); // += 8 * ...
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(fours),  2)); // += 4 * ...
    total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(twos),   1)); // += 2 * ...
    total = _mm256_add_epi64(total, popcount(ones));

    for(; i < size; i++)
        total = _mm256_add_epi64(total, popcount(data[i]));


    return static_cast<uint64_t>(_mm256_extract_epi64(total, 0))
           + static_cast<uint64_t>(_mm256_extract_epi64(total, 1))
           + static_cast<uint64_t>(_mm256_extract_epi64(total, 2))
           + static_cast<uint64_t>(_mm256_extract_epi64(total, 3));
}
} // popcnt

uint64_t sum(const uint8_t* data, const size_t size)
{
    return popcnt::popcnt((const __m256i*) data, size / 32);
}

int main() {
    // avx256 does 8 floats at once at each register.
    xt::xarray<float> arr;
    arr.resize({256});
    int _sign = 1;
    for(int i = 0; i < 256; i++){
        arr(i) = i * _sign;
        _sign *= -1;
    }
    for(auto i : arr){
        std::cout << i << " ";
    }
    std::cout << std::endl;
    float* data = arr.data() + arr.data_offset();
    xtl::xdynamic_bitset<std::uint8_t> bitset;
    bitset.resize({256});

    // Move 8 floats at a time
    sign(arr.data(), bitset.data(), arr.size());

    for(auto i = 0; i < bitset.size(); i++){
        std::cout << bitset[i] << " ";
    }
    std::cout << "\n\n\n" << std::endl;


    xtl::xdynamic_bitset<std::uint8_t> result;
    result.resize({256});
    // Perform XNOR on itself - it should be 0.
    xnor(bitset.data(), bitset.data(), result.data());

    std::cout << sum((const std::uint8_t *)result.data(), result.size()/NUM_BITS) << std::endl;

    return 0;
}