/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSIMD_SSE_INT64_HPP
#define XSIMD_SSE_INT64_HPP

#include <cstdint>

#include "xsimd_base.hpp"

namespace xsimd
{

    /**************************
     * batch_bool<int64_t, 2> *
     **************************/

    template <>
    struct simd_batch_traits<batch_bool<int64_t, 2>>
    {
        using value_type = int64_t;
        static constexpr std::size_t size = 2;
        using batch_type = batch<int64_t, 2>;
        static constexpr std::size_t align = 16;
    };

    template <>
    class batch_bool<int64_t, 2> : public simd_batch_bool<batch_bool<int64_t, 2>>
    {
    public:

        batch_bool();
        explicit batch_bool(bool b);
        batch_bool(bool b0, bool b1);
        batch_bool(const __m128i& rhs);
        batch_bool& operator=(const __m128i& rhs);

        operator __m128i() const;

        bool operator[](std::size_t index) const;

    private:

        __m128i m_value;
    };

    /*********************
     * batch<int64_t, 2> *
     *********************/

    template <>
    struct simd_batch_traits<batch<int64_t, 2>>
    {
        using value_type = int64_t;
        static constexpr std::size_t size = 2;
        using batch_bool_type = batch_bool<int64_t, 2>;
        static constexpr std::size_t align = 16;
    };

    template <>
    class batch<int64_t, 2> : public simd_batch<batch<int64_t, 2>>
    {
    public:

        using self_type = batch<int64_t, 2>;
        using base_type = simd_batch<self_type>;

        batch();
        explicit batch(int64_t i);
        batch(int64_t i0, int64_t i1);
        explicit batch(const int64_t* src);
        batch(const int64_t* src, aligned_mode);
        batch(const int64_t* src, unaligned_mode);
        batch(const __m128i& rhs);
        batch& operator=(const __m128i& rhs);

        operator __m128i() const;

        batch& load_aligned(const int64_t* src);
        batch& load_unaligned(const int64_t* src);

        batch& load_aligned(const int32_t* src);
        batch& load_unaligned(const int32_t* src);

        batch& load_aligned(const float* src);
        batch& load_unaligned(const float* src);

        batch& load_aligned(const double* src);
        batch& load_unaligned(const double* src);

        batch& load_aligned(const int8_t* src);
        batch& load_unaligned(const int8_t* src);

        batch& load_aligned(const uint8_t* src);
        batch& load_unaligned(const uint8_t* src);

        void store_aligned(int64_t* dst) const;
        void store_unaligned(int64_t* dst) const;

        void store_aligned(int32_t* dst) const;
        void store_unaligned(int32_t* dst) const;

        void store_aligned(float* dst) const;
        void store_unaligned(float* dst) const;

        void store_aligned(double* dst) const;
        void store_unaligned(double* dst) const;

        void store_aligned(int8_t* dst) const;
        void store_unaligned(int8_t* dst) const;

        void store_aligned(uint8_t* dst) const;
        void store_unaligned(uint8_t* dst) const;

        using base_type::load_aligned;
        using base_type::load_unaligned;
        using base_type::store_aligned;
        using base_type::store_unaligned;

        int64_t operator[](std::size_t index) const;

    private:

        __m128i m_value;
    };

    batch<int64_t, 2> operator<<(const batch<int64_t, 2>& lhs, int32_t rhs);
    batch<int64_t, 2> operator>>(const batch<int64_t, 2>& lhs, int32_t rhs);

    /********************
     * helper functions *
     ********************/

    namespace detail
    {
        inline __m128i cmpeq_epi64_sse2(__m128i lhs, __m128i rhs)
        {
            __m128i tmp1 = _mm_cmpeq_epi32(lhs, rhs);
            __m128i tmp2 = _mm_shuffle_epi32(tmp1, 0xB1);
            __m128i tmp3 = _mm_and_si128(tmp1, tmp2);
            __m128i tmp4 = _mm_srai_epi32(tmp3, 31);
            return _mm_shuffle_epi32(tmp4, 0xF5);
        }
    }

    /*****************************************
     * batch_bool<int64_t, 2> implementation *
     *****************************************/

    inline batch_bool<int64_t, 2>::batch_bool()
    {
    }

    inline batch_bool<int64_t, 2>::batch_bool(bool b)
        : m_value(_mm_set1_epi64x(-(int64_t)b))
    {
    }

    inline batch_bool<int64_t, 2>::batch_bool(bool b0, bool b1)
        : m_value(_mm_set_epi64x(-(int64_t)b1, -(int64_t)b0))
    {
    }

    inline batch_bool<int64_t, 2>::batch_bool(const __m128i& rhs)
        : m_value(rhs)
    {
    }

    inline batch_bool<int64_t, 2>& batch_bool<int64_t, 2>::operator=(const __m128i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch_bool<int64_t, 2>::operator __m128i() const
    {
        return m_value;
    }

    inline bool batch_bool<int64_t, 2>::operator[](std::size_t index) const
    {
        alignas(16) int64_t x[2];
        _mm_store_si128((__m128i*)x, m_value);
        return static_cast<bool>(x[index & 1]);
    }

    namespace detail
    {
        template <>
        struct batch_bool_kernel<int64_t, 2>
        {
            using batch_type = batch_bool<int64_t, 2>;

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_and_si128(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_or_si128(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_xor_si128(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm_xor_si128(rhs, _mm_set1_epi32(-1));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_andnot_si128(lhs, rhs);
            }

            static batch_type equal(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return _mm_cmpeq_epi64(lhs, rhs);
#else
                return detail::cmpeq_epi64_sse2(lhs, rhs);
#endif
            }

            static batch_type not_equal(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static bool all(const batch_type& rhs)
            {
                return _mm_movemask_epi8(rhs) == 0xFFFF;
            }

            static bool any(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return !_mm_testz_si128(rhs, rhs);
#else
                return _mm_movemask_epi8(rhs) != 0;
#endif
            }
        };
    }

    /************************************
     * batch<int64_t, 2> implementation *
     ************************************/

    inline batch<int64_t, 2>::batch()
    {
    }

    inline batch<int64_t, 2>::batch(int64_t i)
        : m_value(_mm_set1_epi64x(i))
    {
    }

    inline batch<int64_t, 2>::batch(int64_t i0, int64_t i1)
        : m_value(_mm_set_epi64x(i1, i0))
    {
    }

    inline batch<int64_t, 2>::batch(const int64_t* src)
        : m_value(_mm_loadu_si128((__m128i const*)src))
    {
    }

    inline batch<int64_t, 2>::batch(const int64_t* src, aligned_mode)
        : m_value(_mm_load_si128((__m128i const*)src))
    {
    }

    inline batch<int64_t, 2>::batch(const int64_t* src, unaligned_mode)
        : m_value(_mm_loadu_si128((__m128i const*)src))
    {
    }

    inline batch<int64_t, 2>::batch(const __m128i& rhs)
        : m_value(rhs)
    {
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::operator=(const __m128i& rhs)
    {
        m_value = rhs;
        return *this;
    }

    inline batch<int64_t, 2>::operator __m128i() const
    {
        return m_value;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const int64_t* src)
    {
        m_value = _mm_load_si128((__m128i const*)src);
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const int64_t* src)
    {
        m_value = _mm_loadu_si128((__m128i const*)src);
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const int32_t* src)
    {
        alignas(16) int64_t tmp[2];
        tmp[0] = int64_t(src[0]);
        tmp[1] = int64_t(src[1]);
        m_value = load_aligned(tmp);
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const int32_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const float* src)
    {
        alignas(16) int64_t tmp[2];
        tmp[0] = static_cast<int64_t>(src[0]);
        tmp[1] = static_cast<int64_t>(src[1]);
        m_value = load_aligned(tmp);
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const float* src)
    {
        return load_aligned(src);
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const double* src)
    {
        alignas(16) int64_t tmp[2];
        tmp[0] = static_cast<int64_t>(src[0]);
        tmp[1] = static_cast<int64_t>(src[1]);
        m_value = load_aligned(tmp);
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const double* src)
    {
        return load_aligned(src);
    }


    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const int8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        m_value = _mm_cvtepi8_epi64(tmp);
#else
        __m128i mask = _mm_cmplt_epi8(tmp, _mm_set1_epi8(0));
        __m128i tmp1 = _mm_unpacklo_epi8(tmp, mask);
        mask = _mm_cmplt_epi16(tmp1, _mm_set1_epi16(0));
        __m128i tmp2 = _mm_unpacklo_epi16(tmp1, mask);
        mask = _mm_cmplt_epi32(tmp2, _mm_set1_epi32(0));
        m_value = _mm_unpacklo_epi32(tmp2, mask);
#endif
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const int8_t* src)
    {
        return load_aligned(src);
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_aligned(const uint8_t* src)
    {
        __m128i tmp = _mm_loadl_epi64((const __m128i*)src);
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
        m_value = _mm_cvtepu8_epi64(tmp);
#else
        __m128i tmp1 = _mm_unpacklo_epi8(tmp, _mm_set1_epi8(0));
        __m128i tmp2 = _mm_unpacklo_epi16(tmp, _mm_set1_epi16(0));
        m_value = _mm_unpacklo_epi32(tmp, _mm_set1_epi32(0));
#endif
        return *this;
    }

    inline batch<int64_t, 2>& batch<int64_t, 2>::load_unaligned(const uint8_t* src)
    {
        return load_aligned(src);
    }

    inline void batch<int64_t, 2>::store_aligned(int64_t* dst) const
    {
        _mm_store_si128((__m128i*)dst, m_value);
    }

    inline void batch<int64_t, 2>::store_unaligned(int64_t* dst) const
    {
        _mm_storeu_si128((__m128i*)dst, m_value);
    }

    inline void batch<int64_t, 2>::store_aligned(int32_t* dst) const
    {
        alignas(16) int64_t tmp[2];
        store_aligned(tmp);
        dst[0] = static_cast<int32_t>(tmp[0]);
        dst[1] = static_cast<int32_t>(tmp[1]);
    }

    inline void batch<int64_t, 2>::store_unaligned(int32_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 2>::store_aligned(float* dst) const
    {
        alignas(16) int64_t tmp[2];
        store_aligned(tmp);
        dst[0] = float(tmp[0]);
        dst[1] = float(tmp[1]);
    }

    inline void batch<int64_t, 2>::store_unaligned(float* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 2>::store_aligned(double* dst) const
    {
        alignas(16) int64_t tmp[2];
        store_aligned(tmp);
        dst[0] = double(tmp[0]);
        dst[1] = double(tmp[1]);
    }

    inline void batch<int64_t, 2>::store_unaligned(double* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 2>::store_aligned(int8_t* dst) const
    {
        alignas(16) int64_t tmp[2];
        store_aligned(tmp);
        dst[0] = static_cast<char>(tmp[0]);
        dst[1] = static_cast<char>(tmp[1]);
    }

    inline void batch<int64_t, 2>::store_unaligned(int8_t* dst) const
    {
        store_aligned(dst);
    }

    inline void batch<int64_t, 2>::store_aligned(uint8_t* dst) const
    {
        alignas(16) int64_t tmp[2];
        store_aligned(tmp);
        dst[0] = static_cast<unsigned char>(tmp[0]);
        dst[1] = static_cast<unsigned char>(tmp[1]);
    }

    inline void batch<int64_t, 2>::store_unaligned(uint8_t* dst) const
    {
        store_aligned(dst);
    }

    inline int64_t batch<int64_t, 2>::operator[](std::size_t index) const
    {
        alignas(16) int64_t x[2];
        store_aligned(x);
        return x[index & 1];
    }

    namespace detail
    {
        template <>
        struct batch_kernel<int64_t, 2>
        {
            using batch_type = batch<int64_t, 2>;
            using value_type = int64_t;
            using batch_bool_type = batch_bool<int64_t, 2>;

            static batch_type neg(const batch_type& rhs)
            {
                return _mm_sub_epi64(_mm_setzero_si128(), rhs);
            }

            static batch_type add(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_add_epi64(lhs, rhs);
            }

            static batch_type sub(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_sub_epi64(lhs, rhs);
            }

            static batch_type mul(const batch_type& lhs, const batch_type& rhs)
            {
                alignas(16) int64_t slhs[2], srhs[2];
                lhs.store_aligned(slhs);
                rhs.store_aligned(srhs);
                return batch<int64_t, 2>(slhs[0] * srhs[0], slhs[1] * srhs[1]);
            }

            static batch_type div(const batch_type& lhs, const batch_type& rhs)
            {
#if defined(XSIMD_FAST_INTEGER_DIVISION)
                __m128d dlhs = _mm_setr_pd(static_cast<double>(lhs[0]), static_cast<double>(lhs[1]));
                __m128d drhs = _mm_setr_pd(static_cast<double>(rhs[0]), static_cast<double>(rhs[1]));
                __m128i tmp = _mm_cvttpd_epi32(_mm_div_pd(dlhs, drhs));
                using batch_int = batch<int64_t, 2>;
                return _mm_unpacklo_epi32(tmp, batch_int(tmp) < batch_int(int64_t(0)));
#else
                alignas(16) int64_t tmp_lhs[2], tmp_rhs[2], tmp_res[2];
                lhs.store_aligned(tmp_lhs);
                rhs.store_aligned(tmp_rhs);
                tmp_res[0] = tmp_lhs[0] / tmp_rhs[0];
                tmp_res[1] = tmp_lhs[1] / tmp_rhs[1];
                return batch_type(&tmp_res[0], aligned_mode());
#endif

            }

            static batch_bool_type eq(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return _mm_cmpeq_epi64(lhs, rhs);
#else
                return detail::cmpeq_epi64_sse2(lhs, rhs);
#endif
            }

            static batch_bool_type neq(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(lhs == rhs);
            }

            static batch_bool_type lt(const batch_type& lhs, const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_2_VERSION
                return _mm_cmpgt_epi64(rhs, lhs);
#else
                __m128i tmp1 = _mm_sub_epi64(lhs, rhs);
                __m128i tmp2 = _mm_xor_si128(lhs, rhs);
                __m128i tmp3 = _mm_andnot_si128(rhs, lhs);
                __m128i tmp4 = _mm_andnot_si128(tmp2, tmp1);
                __m128i tmp5 = _mm_or_si128(tmp3, tmp4);
                __m128i tmp6 = _mm_srai_epi32(tmp5, 31);
                return _mm_shuffle_epi32(tmp6, 0xF5);
#endif
            }

            static batch_bool_type lte(const batch_type& lhs, const batch_type& rhs)
            {
                return ~(rhs < lhs);
            }

            static batch_type bitwise_and(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_and_si128(lhs, rhs);
            }

            static batch_type bitwise_or(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_or_si128(lhs, rhs);
            }

            static batch_type bitwise_xor(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_xor_si128(lhs, rhs);
            }

            static batch_type bitwise_not(const batch_type& rhs)
            {
                return _mm_xor_si128(rhs, _mm_set1_epi32(-1));
            }

            static batch_type bitwise_andnot(const batch_type& lhs, const batch_type& rhs)
            {
                return _mm_andnot_si128(lhs, rhs);
            }

            static batch_type min(const batch_type& lhs, const batch_type& rhs)
            {
                return select(lhs < rhs, lhs, rhs);
            }

            static batch_type max(const batch_type& lhs, const batch_type& rhs)
            {
                return select(lhs > rhs, lhs, rhs);
            }

            static batch_type abs(const batch_type& rhs)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_2_VERSION
                __m128i sign = _mm_cmpgt_epi64(_mm_setzero_si128(), rhs);
#else
                __m128i signh = _mm_srai_epi32(rhs, 31);
                __m128i sign = _mm_shuffle_epi32(signh, 0xF5);
#endif
                __m128i inv = _mm_xor_si128(rhs, sign);
                return _mm_sub_epi64(inv, sign);
            }

            static batch_type fma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return x * y + z;
            }

            static batch_type fms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return x * y - z;
            }

            static batch_type fnma(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return -x * y + z;
            }

            static batch_type fnms(const batch_type& x, const batch_type& y, const batch_type& z)
            {
                return -x * y - z;
            }

            static value_type hadd(const batch_type& rhs)
            {
                __m128i tmp1 = _mm_shuffle_epi32(rhs, 0x0E);
                __m128i tmp2 = _mm_add_epi64(rhs, tmp1);
#if defined(__x86_64__)
                return _mm_cvtsi128_si64(tmp2);
#else
                union {
                    int64_t i;
                    __m128i m;
                } u;
                _mm_storel_epi64(&u.m, tmp2);
                return u.i;
#endif
            }

            static batch_type select(const batch_bool_type& cond, const batch_type& a, const batch_type& b)
            {
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE4_1_VERSION
                return _mm_blendv_epi8(b, a, cond);
#else
                return _mm_or_si128(_mm_and_si128(cond, a), _mm_andnot_si128(cond, b));
#endif
            }
        };
    }

    inline batch<int64_t, 2> operator<<(const batch<int64_t, 2>& lhs, int32_t rhs)
    {
        return _mm_slli_epi64(lhs, rhs);
    }

    inline batch<int64_t, 2> operator>>(const batch<int64_t, 2>& lhs, int32_t rhs)
    {
        return _mm_srli_epi64(lhs, rhs);
    }
}

#endif
