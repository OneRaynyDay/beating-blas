/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_EXPRESSION_HPP
#define XTENSOR_EXPRESSION_HPP

#include <cstddef>
#include <type_traits>
#include <vector>

#include <xtl/xclosure.hpp>
#include <xtl/xmeta_utils.hpp>
#include <xtl/xtype_traits.hpp>

#include "xlayout.hpp"
#include "xshape.hpp"
#include "xutils.hpp"

namespace xt
{

    template <class E>
    class xshared_expression;

    /***************************
     * xexpression declaration *
     ***************************/

    /**
     * @class xexpression
     * @brief Base class for xexpressions
     *
     * The xexpression class is the base class for all classes representing an expression
     * that can be evaluated to a multidimensional container with tensor semantic.
     * Functions that can apply to any xexpression regardless of its specific type should take a
     * xexpression argument.
     *
     * \tparam E The derived type.
     *
     */
    template <class D>
    class xexpression
    {
    public:

        using derived_type = D;

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const & noexcept;
        derived_type derived_cast() && noexcept;

    protected:

        xexpression() = default;
        ~xexpression() = default;

        xexpression(const xexpression&) = default;
        xexpression& operator=(const xexpression&) = default;

        xexpression(xexpression&&) = default;
        xexpression& operator=(xexpression&&) = default;
    };

    /******************************
     * xexpression implementation *
     ******************************/

    /**
     * @name Downcast functions
     */
    //@{
    /**
     * Returns a reference to the actual derived type of the xexpression.
     */
    template <class D>
    inline auto xexpression<D>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    /**
     * Returns a constant reference to the actual derived type of the xexpression.
     */
    template <class D>
    inline auto xexpression<D>::derived_cast() const & noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /**
     * Returns a constant reference to the actual derived type of the xexpression.
     */
    template <class D>
    inline auto xexpression<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }
    //@}

    /* is_crtp_base_of<B, E>
    * Resembles std::is_base_of, but adresses the problem of whether _some_ instatntiation
    * of a CRTP templated class B is a base of class E. A CRTP templated class is correctly
    * templated with the most derived type in the CRTP hierarchy. Using this assumption,
    * this implementation deals with either CRTP final classes (checks for inheritance
    * with E as the CRTP parameter of B) or CRTP base classes (which are singly templated
    * by the most derived class, and that's pulled out to use as a templete parameter for B).
    */

    namespace detail
    {
        template <template<class> class B, class E>
        struct is_crtp_base_of_impl : std::is_base_of<B<E>, E> {};

        template <template<class> class B, class E, template<class> class F>
        struct is_crtp_base_of_impl<B, F<E>> :
        xtl::disjunction< std::is_base_of<B<E>, F<E>>, std::is_base_of<B<F<E>>, F<E>>> {};
    }

    template <template<class> class B, class E>
    using is_crtp_base_of = detail::is_crtp_base_of_impl<B, std::decay_t<E>>;

    template <class E>
    using is_xexpression = is_crtp_base_of<xexpression, E>;

    template <class E, class R = void>
    using enable_xexpression = typename std::enable_if<is_xexpression<E>::value, R>::type;

    template <class E, class R = void>
    using disable_xexpression = typename std::enable_if<!is_xexpression<E>::value, R>::type;

    template <class... E>
    using has_xexpression = xtl::disjunction<is_xexpression<E>...>;

    /************
     * xclosure *
     ************/

    template <class T>
    class xscalar;

    template <class E, class EN = void>
    struct xclosure
    {
        using type = xtl::closure_type_t<E>;
    };

    template <class E>
    struct xclosure<xshared_expression<E>, std::enable_if_t<true>>
    {
        using type = xshared_expression<E>; // force copy
    };

    template <class E>
    struct xclosure<E, disable_xexpression<std::decay_t<E>>>
    {
        using type = xscalar<xtl::closure_type_t<E>>;
    };

    template <class E>
    using xclosure_t = typename xclosure<E>::type;

    template <class E, class EN = void>
    struct const_xclosure
    {
        using type = xtl::const_closure_type_t<E>;
    };

    template <class E>
    struct const_xclosure<E, disable_xexpression<std::decay_t<E>>>
    {
        using type = xscalar<xtl::const_closure_type_t<E>>;
    };

    template <class E>
    struct const_xclosure<xshared_expression<E>&, std::enable_if_t<true>>
    {
        using type = xshared_expression<E>; // force copy
    };

    template <class E>
    using const_xclosure_t = typename const_xclosure<E>::type;

    /***************
     * xvalue_type *
     ***************/

    namespace detail
    {
        template <class E, class enable = void>
        struct xvalue_type_impl
        {
            using type = E;
        };

        template <class E>
        struct xvalue_type_impl<E, std::enable_if_t<is_xexpression<E>::value>>
        {
            using type = typename E::value_type;
        };
    }

    template <class E>
    using xvalue_type = detail::xvalue_type_impl<E>;

    template <class E>
    using xvalue_type_t = typename xvalue_type<E>::type;

    /*************************
     * expression tag system *
     *************************/

    struct xscalar_expression_tag
    {
    };

    struct xtensor_expression_tag
    {
    };

    struct xoptional_expression_tag
    {
    };

    namespace detail
    {
        template <class E, class = void_t<int>>
        struct get_expression_tag
        {
            using type = xtensor_expression_tag;
        };

        template <class E>
        struct get_expression_tag<E, void_t<typename std::decay_t<E>::expression_tag>>
        {
            using type = typename std::decay_t<E>::expression_tag;
        };

        template <class E>
        using get_expression_tag_t = typename get_expression_tag<E>::type;

        template <class... T>
        struct expression_tag_and;

        template <class T>
        struct expression_tag_and<T>
        {
            using type = T;
        };

        template <class T>
        struct expression_tag_and<T, T>
        {
            using type = T;
        };

        template <>
        struct expression_tag_and<xscalar_expression_tag, xscalar_expression_tag>
        {
            using type = xscalar_expression_tag;
        };

        template <class T>
        struct expression_tag_and<xscalar_expression_tag, T>
        {
            using type = T;
        };

        template <class T>
        struct expression_tag_and<T, xscalar_expression_tag>
            : expression_tag_and<xscalar_expression_tag, T>
        {
        };

        template <>
        struct expression_tag_and<xtensor_expression_tag, xoptional_expression_tag>
        {
            using type = xoptional_expression_tag;
        };

        template <>
        struct expression_tag_and<xoptional_expression_tag, xtensor_expression_tag>
            : expression_tag_and<xtensor_expression_tag, xoptional_expression_tag>
        {
        };

        template <class T1, class... T>
        struct expression_tag_and<T1, T...>
            : expression_tag_and<T1, typename expression_tag_and<T...>::type>
        {
        };

        template <class... T>
        using expression_tag_and_t = typename expression_tag_and<T...>::type;
    }

    template <class... T>
    struct xexpression_tag
    {
        using type = detail::expression_tag_and_t<detail::get_expression_tag_t<std::decay_t<const_xclosure_t<T>>>...>;
    };

    template <class... T>
    using xexpression_tag_t = typename xexpression_tag<T...>::type;

    template <class E>
    struct is_xtensor_expression : std::is_same<xexpression_tag_t<E>, xtensor_expression_tag>
    {
    };

    template <class E>
    struct is_xoptional_expression : std::is_same<xexpression_tag_t<E>, xoptional_expression_tag>
    {
    };

    /********************************
     * xoptional_comparable concept *
     ********************************/

    template <class... E>
    struct xoptional_comparable : xtl::conjunction<xtl::disjunction<is_xtensor_expression<E>,
                                                                    is_xoptional_expression<E>
                                                                   >...
                                                  >
    {
    };

#define XTENSOR_FORWARD_METHOD(name)           \
    auto name() const                          \
        -> decltype(std::declval<E>().name())  \
    {                                          \
        return m_ptr->name();                  \
    }

    namespace detail
    {
        template <class E>
        struct expr_strides_type
        {
            using type = typename E::strides_type;
        };

        template <class E>
        struct expr_inner_strides_type
        {
            using type = typename E::inner_strides_type;
        };

        template <class E>
        struct expr_backstrides_type
        {
            using type = typename E::backstrides_type;
        };

        template <class E>
        struct expr_inner_backstrides_type
        {
            using type = typename E::inner_backstrides_type;
        };
    }

    /**
     * @class xshared_expression
     * @brief Shared xexpressions
     *
     * Due to C++ lifetime constraints it's sometimes necessary to create shared
     * expressions (akin to a shared pointer).
     *
     * For example, when a temporary expression needs to be used twice in another
     * expression, shared expressions can come to the rescue:
     *
     * \code{.cpp}
     * template <class E>
     * auto cos_plus_sin(xexpression<E>&& expr)
     * {
     *     // THIS IS WRONG: forwarding rvalue twice not permitted!
     *     // return xt::sin(std::forward<E>(expr)) + xt::cos(std::forward<E>(expr));
     *     // THIS IS WRONG TOO: because second `expr` is taken as reference (which will be invalid)
     *     // return xt::sin(std::forward<E>(expr)) + xt::cos(expr)
     *     auto shared_expr = xt::make_xshared(std::forward<E>(expr));
     *     auto result = xt::sin(shared_expr) + xt::cos(shared_expr);
     *     std::cout << shared_expr.use_count() << std::endl; // Will print 3 because used twice in expression
     *     return result; // all valid because expr lifetime managed by xshared_expression / shared_ptr.
     * }
     * \endcode
     */
    template <class E>
    class xshared_expression
        : public xexpression<xshared_expression<E>>
    {
    public:

        using base_class = xexpression<xshared_expression<E>>;

        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using const_reference = typename E::const_reference;
        using pointer = typename E::pointer;
        using const_pointer = typename E::const_pointer;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;

        using inner_shape_type = typename E::inner_shape_type;
        using shape_type = typename E::shape_type;

        using strides_type = xtl::mpl::eval_if_t<has_strides<E>,
                                                 detail::expr_strides_type<E>,
                                                 get_strides_type<shape_type>>;
        using backstrides_type = xtl::mpl::eval_if_t<has_strides<E>,
                                                     detail::expr_backstrides_type<E>,
                                                     get_strides_type<shape_type>>;
        using inner_strides_type = xtl::mpl::eval_if_t<has_strides<E>,
                                                       detail::expr_inner_strides_type<E>,
                                                       get_strides_type<shape_type>>;
        using inner_backstrides_type = xtl::mpl::eval_if_t<has_strides<E>,
                                                           detail::expr_inner_backstrides_type<E>,
                                                           get_strides_type<shape_type>>;

        using stepper = typename E::stepper;
        using const_stepper = typename E::const_stepper;

        using storage_iterator = typename E::storage_iterator;
        using const_storage_iterator = typename E::const_storage_iterator;

        static constexpr layout_type static_layout = E::static_layout;
        static constexpr bool contiguous_layout = static_layout != layout_type::dynamic;

        explicit xshared_expression(std::shared_ptr<E>&& ptr);
        long use_count() const noexcept;

        template <class... Args>
        auto operator()(Args... args)
            -> decltype(std::declval<E>()(args...))
        {
            return m_ptr->operator()(args...);
        }

        XTENSOR_FORWARD_METHOD(shape);
        XTENSOR_FORWARD_METHOD(dimension);
        XTENSOR_FORWARD_METHOD(size);
        XTENSOR_FORWARD_METHOD(begin);
        XTENSOR_FORWARD_METHOD(cbegin);
        XTENSOR_FORWARD_METHOD(storage_begin);
        XTENSOR_FORWARD_METHOD(storage_cbegin);
        XTENSOR_FORWARD_METHOD(storage_end);
        XTENSOR_FORWARD_METHOD(storage_cend);
        XTENSOR_FORWARD_METHOD(layout);

        template <class T = E>
        std::enable_if_t<has_strides<T>::value, const inner_strides_type&>
        strides() const
        {
            return m_ptr->strides();
        }

        template <class T = E>
        std::enable_if_t<has_strides<T>::value, const inner_strides_type&>
        backstrides() const
        {
            return m_ptr->backstrides();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, pointer>
        data() noexcept
        {
            return m_ptr->data();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, pointer>
        data() const noexcept
        {
            return m_ptr->data();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, size_type>
        data_offset() const noexcept
        {
            return m_ptr->data_offset();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, typename T::storage_type&>
        storage() noexcept
        {
            return m_ptr->storage();
        }

        template <class T = E>
        std::enable_if_t<has_data_interface<T>::value, const typename T::storage_type&>
        storage() const noexcept
        {
            return m_ptr->storage();
        }

        template <class It>
        auto element(It first, It last) {
            return m_ptr->element(first, last);
        }

        template <class It>
        auto element(It first, It last) const {
            return m_ptr->element(first, last);
        }

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const
        {
            return m_ptr->broadcast_shape(shape, reuse_cache);
        }

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept
        {
            return m_ptr->is_trivial_broadcast(strides);
        }

        template <class S>
        auto stepper_begin(const S& shape) noexcept
            -> decltype(std::declval<E>().stepper_begin(shape))
        {
            return m_ptr->stepper_begin(shape);
        }

        template <class S>
        auto stepper_end(const S& shape, layout_type l) noexcept
            -> decltype(std::declval<E>().stepper_end(shape, l))
        {
            return m_ptr->stepper_end(shape, l);
        }

        template <class S>
        auto stepper_begin(const S& shape) const noexcept
            -> decltype(std::declval<const E>().stepper_begin(shape))
        {
            return static_cast<const E*>(m_ptr.get())->stepper_begin(shape);
        }
        template <class S>
        auto stepper_end(const S& shape, layout_type l) const noexcept
            -> decltype(std::declval<const E>().stepper_end(shape, l))
        {
            return static_cast<const E*>(m_ptr.get())->stepper_end(shape, l);
        }

    private:

        std::shared_ptr<E> m_ptr;
    };

    /**
     * Constructor for xshared expression (note: usually the free function
     * `make_xshared` is recommended).
     *
     * @param ptr shared ptr that contains the expression
     * @sa make_xshared
     */
    template <class E>
    inline xshared_expression<E>::xshared_expression(std::shared_ptr<E>&& ptr)
        : m_ptr(std::move(ptr))
    {
    }

    /**
     * Return the number of times this expression is referenced.
     * Internally calls the use_count() function of the std::shared_ptr.
     */
    template <class E>
    inline long xshared_expression<E>::use_count() const noexcept
    {
        return m_ptr.use_count();
    }

    /**
     * Helper function to create shared expression from any xexpression
     *
     * @param expr rvalue expression that will be shared
     * @return xshared expression
     */
    template <class E>
    auto make_xshared(xexpression<E>&& expr)
    {
        return xshared_expression<E>(std::make_shared<E>(std::move(expr).derived_cast()));
    }

#undef XTENSOR_FORWARD_METHOD

}

#endif
