/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_DYNAMIC_VIEW_HPP
#define XTENSOR_DYNAMIC_VIEW_HPP

#include <xtl/xsequence.hpp>
#include <xtl/xvariant.hpp>

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xlayout.hpp"
#include "xsemantic.hpp"
#include "xstrided_view_base.hpp"

namespace xt
{

    template <class CT, class S, layout_type L, class FST>
    class xdynamic_view;

    template <class CT, class S, layout_type L, class FST>
    struct xcontainer_inner_types<xdynamic_view<CT, S, L, FST>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = xarray<std::decay_t<typename xexpression_type::value_type>>;
    };

    template <class CT, class S, layout_type L, class FST>
    struct xiterable_inner_types<xdynamic_view<CT, S, L, FST>>
    {
        using inner_shape_type = S;
        using inner_strides_type = inner_shape_type;
        using inner_backstrides_type = inner_shape_type;

        // TODO: implement efficient stepper specific to the dynamic_view
        using const_stepper = xindexed_stepper<const xdynamic_view<CT, S, L, FST>, true>;
        using stepper = xindexed_stepper<xdynamic_view<CT, S, L, FST>, false>;
    };

    /*****************
     * xdynamic_view *
     *****************/

    namespace detail
    {
        template <class T>
        class xfake_slice;
    }

    template <class CT, class S, layout_type L = layout_type::dynamic, class FST = typename detail::flat_storage_type<CT>::type>
    class xdynamic_view : public xview_semantic<xdynamic_view<CT, S, L, FST>>,
                          public xiterable<xdynamic_view<CT, S, L, FST>>,
                          private xstrided_view_base<CT, S, L, FST>
    {
    public:

        using self_type = xdynamic_view<CT, S, L, FST>;
        using base_type = xstrided_view_base<CT, S, L, FST>;
        using semantic_base = xview_semantic<self_type>;

        using xexpression_type = typename base_type::xexpression_type;
        using base_type::is_const;

        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;

        using inner_storage_type = typename base_type::inner_storage_type;
        using storage_type = typename base_type::storage_type;

        using iterable_base = xiterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;

        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using base_type::static_layout;
        using base_type::contiguous_layout;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        using simd_value_type = xsimd::simd_type<value_type>;
        using strides_vt = typename strides_type::value_type;
        using slice_type = xtl::variant<detail::xfake_slice<strides_vt>, xkeep_slice<strides_vt>, xdrop_slice<strides_vt>>;
        using slice_vector_type = std::vector<slice_type>;

        template <class CTA>
        xdynamic_view(CTA&& e, S&& shape, get_strides_t<S>&& strides, std::size_t offset, layout_type layout,
                      slice_vector_type&& slices, get_strides_t<S>&& adj_strides) noexcept;

        template <class CTA, class FLS>
        xdynamic_view(CTA&& e, S&& shape, get_strides_t<S>&& strides, std::size_t offset, layout_type layout,
                      FLS&& flatten_strides, layout_type flatten_layout,
                      slice_vector_type&& slices, get_strides_t<S>&& adj_strides) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        using base_type::size;
        using base_type::dimension;
        using base_type::shape;
        using base_type::layout;

        // Explicitly deleting strides method to avoid compilers complaining
        // about not being able to call the strides method from xstrided_view_base
        // private base
        const inner_strides_type& strides() const noexcept = delete;

        reference operator()();
        const_reference operator()() const;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        reference at(Args... args);

        template <class... Args>
        const_reference at(Args... args) const;

        template <class... Args>
        reference unchecked(Args... args);

        template <class... Args>
        const_reference unchecked(Args... args) const;

        template <class OS>
        disable_integral_t<OS, reference> operator[](const OS& index);
        template <class I>
        reference operator[](std::initializer_list<I> index);
        reference operator[](size_type i);

        template <class OS>
        disable_integral_t<OS, const_reference> operator[](const OS& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        size_type data_offset() const noexcept;

        using base_type::data;
        using base_type::storage;
        using base_type::expression;
        using base_type::broadcast_shape;
        using base_type::is_trivial_broadcast;

        template <class T>
        void fill(const T& value);

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape, layout_type l);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape, layout_type l) const;

        using container_iterator = std::conditional_t<is_const,
                                                      typename storage_type::const_iterator,
                                                      typename storage_type::iterator>;
        using const_container_iterator = typename storage_type::const_iterator;

    private:

        using offset_type = typename base_type::offset_type;

        slice_vector_type m_slices;
        inner_strides_type m_adj_strides;

        container_iterator data_xbegin() noexcept;
        const_container_iterator data_xbegin() const noexcept;
        container_iterator data_xend(layout_type l) noexcept;
        const_container_iterator data_xend(layout_type l) const noexcept;

        template <class It>
        It data_xbegin_impl(It begin) const noexcept;

        template <class It>
        It data_xend_impl(It end, layout_type l) const noexcept;

        void assign_temporary_impl(temporary_type&& tmp);      

        template <class T,  class... Args>
        offset_type adjust_offset(offset_type offset, T idx, Args... args) const noexcept;
        offset_type adjust_offset(offset_type offset) const noexcept;

        template <class T, class... Args>
        offset_type adjust_offset_impl(offset_type offset, size_type idx_offset, T idx, Args... args) const noexcept;
        offset_type adjust_offset_impl(offset_type offset, size_type idx_offset) const noexcept;

        template <class It>
        offset_type adjust_element_offset(offset_type offset, It first, It last) const noexcept;

        template <class C>
        friend class xstepper;
        friend class xview_semantic<xdynamic_view<CT, S, L, FST>>;
    };

    /**************************
     * xdynamic_view builders *
     **************************/

    template <class T>
    using xdynamic_slice = xtl::variant<
        T,

        xrange_adaptor<placeholders::xtuph, T, T>,
        xrange_adaptor<T, placeholders::xtuph, T>,
        xrange_adaptor<T, T, placeholders::xtuph>,

        xrange_adaptor<T, placeholders::xtuph, placeholders::xtuph>,
        xrange_adaptor<placeholders::xtuph, T, placeholders::xtuph>,
        xrange_adaptor<placeholders::xtuph, placeholders::xtuph, T>,

        xrange_adaptor<T, T, T>,
        xrange_adaptor<placeholders::xtuph, placeholders::xtuph, placeholders::xtuph>,

        xkeep_slice<T>,
        xdrop_slice<T>,

        xall_tag,
        xellipsis_tag,
        xnewaxis_tag
    >;

    using xdynamic_slice_vector = std::vector<xdynamic_slice<std::ptrdiff_t>>;

    template <class E>
    auto dynamic_view(E&& e, const xdynamic_slice_vector& slices);

    /******************************
     * xfake_slice implementation *
     ******************************/

    namespace detail
    {
        template <class T>
        class xfake_slice : public xslice<xfake_slice<T>>
        {
        public:

            using size_type = T;
            using self_type = xfake_slice<T>;

            xfake_slice() = default;

            size_type operator()(size_type /*i*/) const noexcept
            {
                return size_type(0);
            }

            size_type size() const noexcept
            {
                return size_type(1);
            }

            size_type step_size() const noexcept
            {
                return size_type(0);
            }

            size_type step_size(std::size_t /*i*/, std::size_t /*n*/ = 1) const noexcept
            {
                return size_type(0);
            }

            size_type revert_index(std::size_t i) const noexcept
            {
                return i;
            }

            bool contains(size_type /*i*/) const noexcept
            {
                return true;
            }

            bool operator==(const self_type& /*rhs*/) const noexcept
            {
                return true;
            }

            bool operator!=(const self_type& /*rhs*/) const noexcept
            {
                return false;
            }
        };
    }

    /********************************
     * xdynamic_view implementation *
     ********************************/

    template <class CT, class S, layout_type L, class FST>
    template <class CTA>
    inline xdynamic_view<CT, S, L, FST>::xdynamic_view(CTA&& e, S&& shape, get_strides_t<S>&& strides,
                                                       std::size_t offset, layout_type layout,
                                                       slice_vector_type&& slices, get_strides_t<S>&& adj_strides) noexcept
        : base_type(std::forward<CTA>(e), std::move(shape), std::move(strides), offset, layout),
          m_slices(std::move(slices)), m_adj_strides(std::move(adj_strides))
    {
    }

    template <class CT, class S, layout_type L, class FST>
    template <class CTA, class FLS>
    inline xdynamic_view<CT, S, L, FST>::xdynamic_view(CTA&& e, S&& shape, get_strides_t<S>&& strides,
                                                       std::size_t offset, layout_type layout,
                                                       FLS&& flatten_strides, layout_type flatten_layout,
                                                       slice_vector_type&& slices, get_strides_t<S>&& adj_strides) noexcept
        : base_type(std::forward<CTA>(e), std::move(shape), std::move(strides),
                    offset, layout, std::forward<FLS>(flatten_strides), flatten_layout),
          m_slices(std::move(slices)), m_adj_strides(std::move(adj_strides))
    {
    }

    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xdynamic_view<CT, S, L, FST>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xdynamic_view<CT, S, L, FST>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::operator()() -> reference
    {
        return base_type::storage()[data_offset()];
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::operator()() const -> const_reference
    {
        return base_type::storage()[data_offset()];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(base_type::shape(), args...));
        XTENSOR_CHECK_DIMENSION(base_type::shape(), args...);
        offset_type offset = base_type::compute_index(args...);
        offset = adjust_offset(offset, args...);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(base_type::shape(), args...));
        XTENSOR_CHECK_DIMENSION(base_type::shape(), args...);
        offset_type offset = base_type::compute_index(args...);
        offset = adjust_offset(offset, args...);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::at(Args... args) -> reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return this->operator()(args...);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::at(Args... args) const -> const_reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return this->operator()(args...);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::unchecked(Args... args) -> reference
    {
        offset_type offset = base_type::compute_unchecked_index(args...);
        offset = adjust_offset(args...);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::unchecked(Args... args) const -> const_reference
    {
        offset_type offset = base_type::compute_unchecked_index(args...);
        offset = adjust_offset(args...);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class OS>
    inline auto xdynamic_view<CT, S, L, FST>::operator[](const OS& index) -> disable_integral_t<OS, reference>
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class S, layout_type L, class FST>
    template <class I>
    inline auto xdynamic_view<CT, S, L, FST>::operator[](std::initializer_list<I> index) -> reference
    {
        return element(index.begin(), index.end());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class OS>
    inline auto xdynamic_view<CT, S, L, FST>::operator[](const OS& index) const -> disable_integral_t<OS, const_reference>
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class S, layout_type L, class FST>
    template <class I>
    inline auto xdynamic_view<CT, S, L, FST>::operator[](std::initializer_list<I> index) const -> const_reference
    {
        return element(index.begin(), index.end());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto xdynamic_view<CT, S, L, FST>::element(It first, It last) -> reference
    {
        XTENSOR_TRY(check_element_index(base_type::shape(), first, last));
        offset_type offset = base_type::compute_element_index(first, last);
        offset = adjust_element_offset(offset, first, last);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto xdynamic_view<CT, S, L, FST>::element(It first, It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(base_type::shape(), first, last));
        offset_type offset = base_type::compute_element_index(first, last);
        offset = adjust_element_offset(offset, first, last);
        return base_type::storage()[static_cast<size_type>(offset)];
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_offset() const noexcept -> size_type
    {
        size_type offset = base_type::data_offset;
        return offset + m_slices[0](size_type(0)) * m_adj_strides[0];
    }

    template <class CT, class S, layout_type L, class FST>
    template <class T>
    inline void xdynamic_view<CT, S, L, FST>::fill(const T& value)
    {
        return std::fill(this->storage_begin(), this->storage_end(), value);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xdynamic_view<CT, S, L, FST>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xdynamic_view<CT, S, L, FST>::stepper_end(const ST& shape, layout_type /*l*/) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset, true);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xdynamic_view<CT, S, L, FST>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class ST>
    inline auto xdynamic_view<CT, S, L, FST>::stepper_end(const ST& shape, layout_type /*l*/) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_xbegin() noexcept -> container_iterator
    {
        return data_xbegin_impl(this->storage().begin());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_xbegin() const noexcept -> const_container_iterator
    {
        return data_xbegin_impl(this->storage().cbegin());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_xend(layout_type l) noexcept -> container_iterator
    {
        return data_xend_impl(this->storage().begin(), l);
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::data_xend(layout_type l) const noexcept -> const_container_iterator
    {
        return data_xend_impl(this->storage().cbegin(), l);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline It xdynamic_view<CT, S, L, FST>::data_xbegin_impl(It begin) const noexcept
    {
        return begin + static_cast<std::ptrdiff_t>(data_offset());
    }

    // TODO: fix the data_xend implementation and assign_temporary_impl

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline It xdynamic_view<CT, S, L, FST>::data_xend_impl(It begin, layout_type l) const noexcept
    {
        std::ptrdiff_t end_offset = static_cast<std::ptrdiff_t>(std::accumulate(this->backstrides().begin(), this->backstrides().end(), std::size_t(0)));
        return strided_data_end(*this, begin + std::ptrdiff_t(data_offset()) + end_offset + 1, l);
    }

    template <class CT, class S, layout_type L, class FST>
    inline void xdynamic_view<CT, S, L, FST>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->begin());
    }

    template <class CT, class S, layout_type L, class FST>
    template <class T, class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::adjust_offset(offset_type offset, T idx, Args... args) const noexcept -> offset_type
    {
        constexpr size_type nb_args = sizeof...(Args) + 1;
        size_type dim = base_type::dimension();
        offset_type res = nb_args > dim ? adjust_offset(offset, args...) : adjust_offset_impl(offset, dim - nb_args, idx, args...);
        return res;
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::adjust_offset(offset_type offset) const noexcept -> offset_type
    {
        return offset;
    }

    template <class CT, class S, layout_type L, class FST>
    template <class T, class... Args>
    inline auto xdynamic_view<CT, S, L, FST>::adjust_offset_impl(offset_type offset, size_type idx_offset, T idx, Args... args) const noexcept
        -> offset_type
    {
        offset_type sl_offset = xtl::visit([idx](const auto& sl) { return sl(idx); }, m_slices[idx_offset]);
        offset_type res = offset + sl_offset * m_adj_strides[idx_offset];
        return adjust_offset_impl(res, idx_offset + 1, args...);
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xdynamic_view<CT, S, L, FST>::adjust_offset_impl(offset_type offset, size_type) const noexcept
        -> offset_type
    {
        return offset;
    }

    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto xdynamic_view<CT, S, L, FST>::adjust_element_offset(offset_type offset, It first, It last) const noexcept -> offset_type
    {
        auto dst = std::distance(first, last);
        offset_type dim = static_cast<offset_type>(dimension());
        offset_type loop_offset = dst < dim ? dim - dst : offset_type(0);
        offset_type idx_offset = dim < dst ? dst - dim : offset_type(0);
        offset_type res = offset;
        for (offset_type i = loop_offset; i < dim; ++i, ++first)
        {
            offset_type j = static_cast<offset_type>(first[idx_offset]);
            offset_type sl_offset = xtl::visit([j](const auto& sl) { return static_cast<offset_type>(sl(j)); }, m_slices[static_cast<std::size_t>(i)]);
            res += sl_offset * m_adj_strides[static_cast<std::size_t>(i)];
        }
        return res;
    }

    /*****************************************
     * xdynamic_view builders implementation *
     *****************************************/

    namespace detail
    {
        template <class V>
        struct adj_strides_policy
        {
            using slice_vector = V;
            using strides_type = dynamic_shape<std::ptrdiff_t>;

            slice_vector new_slices;
            strides_type new_adj_strides;

        protected:

            inline void resize(std::size_t size)
            {
                new_slices.resize(size);
                new_adj_strides.resize(size);
            }

            inline void set_fake_slice(std::size_t idx)
            {
                new_slices[idx] = xfake_slice<std::ptrdiff_t>();
                new_adj_strides[idx] = std::ptrdiff_t(0);
            }

            template <class ST, class S>
            bool fill_args(const xdynamic_slice_vector& slices, std::size_t sl_idx,
                           std::size_t i, std::size_t old_shape,
                           const ST& old_stride,
                           S& shape, get_strides_t<S>& strides)
            {
                return fill_args_impl<xkeep_slice<std::ptrdiff_t>>(slices, sl_idx, i, old_shape, old_stride, shape, strides)
                    || fill_args_impl<xdrop_slice<std::ptrdiff_t>>(slices, sl_idx, i, old_shape, old_stride, shape, strides);
            }

            template <class SL, class ST, class S>
            bool fill_args_impl(const xdynamic_slice_vector& slices, std::size_t sl_idx,
                                std::size_t i, std::size_t old_shape,
                                const ST& old_stride,
                                S& shape, get_strides_t<S>& strides)
            {
                auto* sl = xtl::get_if<SL>(&slices[sl_idx]);
                if (sl != nullptr)
                {
                    new_slices[i] = *sl;
                    auto& ns = xtl::get<SL>(new_slices[i]);
                    ns.normalize(old_shape);
                    shape[i] = static_cast<std::size_t>(ns.size());
                    strides[i] = std::ptrdiff_t(0);
                    new_adj_strides[i] = static_cast<std::ptrdiff_t>(old_stride);
                }
                return sl != nullptr;
            }
        };
    }

    template <class E>
    inline auto dynamic_view(E&& e, const xdynamic_slice_vector& slices)
    {
        using view_type = xdynamic_view<xclosure_t<E>, dynamic_shape<std::size_t>>;
        using slice_vector = typename view_type::slice_vector_type;
        using policy = detail::adj_strides_policy<slice_vector>;
        detail::strided_view_args<policy> args;
        args.fill_args(e.shape(), detail::get_strides(e), detail::get_offset(e), e.layout(), slices);
        return view_type(std::forward<E>(e), std::move(args.new_shape), std::move(args.new_strides), args.new_offset,
                         args.new_layout, std::move(args.new_slices), std::move(args.new_adj_strides));
    }
}

#endif
