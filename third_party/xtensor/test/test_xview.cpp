/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <algorithm>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"


namespace xt
{
    using std::size_t;
    using view_shape_type = dynamic_shape<size_t>;

    TEST(xview, temporary_type)
    {
        {
            view_shape_type shape = {3, 4};
            xarray<double> a(shape);
            auto view1 = view(a, 1, range(1, 4));
            bool check = std::is_same<xarray<double>, typename xcontainer_inner_types<decltype(view1)>::temporary_type>::value;
            EXPECT_TRUE(check);
        }

        {
            xtensor<double, 2>::shape_type shape = {3, 4};
            xtensor<double, 2> a(shape);
            auto view1 = view(a, 1, range(1, 4));
            bool check1 = std::is_same<xtensor<double, 1>, typename xcontainer_inner_types<decltype(view1)>::temporary_type>::value;
            EXPECT_TRUE(check1);

            auto view2 = view(a, all(), newaxis(), range(1, 4));
            bool check2 = std::is_same<xtensor<double, 3>, typename xcontainer_inner_types<decltype(view2)>::temporary_type>::value;
            EXPECT_TRUE(check2);
        }
    }

    TEST(xview, simple)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, 1, range(1, 4));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(size_t(1), view1.dimension());
        EXPECT_ANY_THROW(view1.at(10));
        EXPECT_ANY_THROW(view1.at(0, 0));

        auto view0 = view(a, 0, range(0, 3));
        EXPECT_EQ(a(0, 0), view0(0));
        EXPECT_EQ(a(0, 1), view0(1));
        EXPECT_EQ(size_t(1), view0.dimension());
        EXPECT_EQ(size_t(3), view0.shape()[0]);

        auto view2 = view(a, range(0, 2), 2);
        EXPECT_EQ(a(0, 2), view2(0));
        EXPECT_EQ(a(1, 2), view2(1));
        EXPECT_EQ(size_t(1), view2.dimension());
        EXPECT_EQ(size_t(2), view2.shape()[0]);

        auto view4 = view(a, 1);
        EXPECT_EQ(size_t(1), view4.dimension());
        EXPECT_EQ(size_t(4), view4.shape()[0]);

        auto view5 = view(view4, 1);
        EXPECT_EQ(size_t(0), view5.dimension());
        EXPECT_EQ(size_t(0), view5.shape().size());

        auto view6 = view(a, 1, all());
        EXPECT_EQ(a(1, 0), view6(0));
        EXPECT_EQ(a(1, 1), view6(1));
        EXPECT_EQ(a(1, 2), view6(2));
        EXPECT_EQ(a(1, 3), view6(3));

        auto view7 = view(a, all(), 2);
        EXPECT_EQ(a(0, 2), view7(0));
        EXPECT_EQ(a(1, 2), view7(1));
        EXPECT_EQ(a(2, 2), view7(2));

        if (a.layout() == layout_type::row_major)
        {
            EXPECT_EQ(a.layout(), view1.layout());
            EXPECT_EQ(layout_type::dynamic, view2.layout());
            EXPECT_EQ(a.layout(), view4.layout());
            EXPECT_EQ(a.layout(), view5.layout());
            EXPECT_EQ(a.layout(), view6.layout());
            EXPECT_EQ(layout_type::dynamic, view7.layout());
        }
        else
        {
            EXPECT_EQ(layout_type::dynamic, view1.layout());
            EXPECT_EQ(a.layout(), view2.layout());
            EXPECT_EQ(layout_type::dynamic, view4.layout());
            // TODO ideally this would return the underlying expression's layout
            // but needs special casing 'view-on-view'
            EXPECT_EQ(layout_type::dynamic, view5.layout());
            EXPECT_EQ(layout_type::dynamic, view6.layout());
            EXPECT_EQ(a.layout(), view7.layout());
        }
    }

    TEST(xview, copy_semantic)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        {
            SCOPED_TRACE("copy constructor");
            auto view1 = view(a, 1, range(1, 4));
            auto view2(view1);
            EXPECT_EQ(a(1, 1), view2(0));
            EXPECT_EQ(a(1, 2), view2(1));
            EXPECT_EQ(size_t(1), view2.dimension());
            if (a.layout() == layout_type::row_major)
            {
                EXPECT_EQ(a.layout(), view2.layout());
            }
            else
            {
                EXPECT_EQ(layout_type::dynamic, view2.layout());
            }
        }

        {
            SCOPED_TRACE("copy assignment operator");
            auto view1 = view(a, 1, range(1, 4));
            auto view2 = view(a, 2, range(0, 3));
            view2 = view1;
            EXPECT_EQ(a(2, 0), a(1, 1));
            EXPECT_EQ(a(2, 1), a(1, 2));
            EXPECT_EQ(a(2, 2), a(1, 3));
        }
    }

    TEST(xview, move_semantic)
    {
        view_shape_type shape = { 3, 4 };
        xarray<double> a(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        {
            SCOPED_TRACE("copy constructor");
            auto view1 = view(a, 1, range(1, 4));
            auto view2(std::move(view1));
            EXPECT_EQ(a(1, 1), view2(0));
            EXPECT_EQ(a(1, 2), view2(1));
            EXPECT_EQ(size_t(1), view2.dimension());
            if (a.layout() == layout_type::row_major)
            {
                EXPECT_EQ(a.layout(), view2.layout());
            }
            else
            {
                EXPECT_EQ(layout_type::dynamic, view2.layout());
            }
        }

        {
            SCOPED_TRACE("copy assignment operator");
            auto view1 = view(a, 1, range(1, 4));
            auto view2 = view(a, 2, range(0, 3));
            view2 = std::move(view1);
            EXPECT_EQ(a(2, 0), a(1, 1));
            EXPECT_EQ(a(2, 1), a(1, 2));
            EXPECT_EQ(a(2, 2), a(1, 3));
        }
    }

    TEST(xview, three_dimensional)
    {
        view_shape_type shape = {3, 4, 2};
        std::vector<double> data = {
            1, 2,
            3, 4,
            5, 6,
            7, 8,

            9, 10,
            11, 12,
            21, 22,
            23, 24,

            25, 26,
            27, 28,
            29, 210,
            211, 212
        };
        xarray<double> a(shape);
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, 1);
        EXPECT_EQ(size_t(2), view1.dimension());
        EXPECT_EQ(a(1, 0, 0), view1(0, 0));
        EXPECT_EQ(a(1, 0, 1), view1(0, 1));
        EXPECT_EQ(a(1, 1, 0), view1(1, 0));
        EXPECT_EQ(a(1, 1, 1), view1(1, 1));
        EXPECT_ANY_THROW(view1.at(10, 10));
        EXPECT_ANY_THROW(view1.at(0, 0, 0));

        std::array<std::size_t, 2> idx = {1, 1};
        EXPECT_EQ(a(1, 1, 1), view1.element(idx.cbegin(), idx.cend()));
    }

    TEST(xview, integral_count)
    {
        size_t squeeze1 = integral_count<size_t, size_t, size_t, xrange<size_t>>();
        EXPECT_EQ(squeeze1, size_t(3));
        size_t squeeze2 = integral_count<size_t, xrange<size_t>, size_t>();
        EXPECT_EQ(squeeze2, size_t(2));
        size_t squeeze3 = integral_count_before<size_t, size_t, size_t, xrange<size_t>>(3);
        EXPECT_EQ(squeeze3, size_t(3));
        size_t squeeze4 = integral_count_before<size_t, xrange<size_t>, size_t>(2);
        EXPECT_EQ(squeeze4, size_t(1));
        size_t squeeze5 = integral_count<xnewaxis<size_t>>();
        EXPECT_EQ(squeeze5, size_t(0));
    }

    TEST(xview, integral_skip)
    {
        size_t index0 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>>(0);
        size_t index1 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>>(1);
        size_t index2 = integral_skip<size_t, xrange<size_t>, size_t, xrange<size_t>>(2);
        EXPECT_EQ(index0, size_t(1));
        EXPECT_EQ(index1, size_t(3));
        EXPECT_EQ(index2, size_t(4));
    }

    TEST(xview, single_newaxis_shape)
    {
        xarray<double> a = {1, 2, 3, 4};
        auto v = view(a, newaxis());
        view_shape_type s = {1, 4};
        EXPECT_EQ(s, v.shape());
    }

    TEST(xview, temporary_view)
    {
        xt::xarray<double> arr1
         {{1.0, 2.0, 3.0},
          {2.0, 5.0, 7.0},
          {2.0, 5.0, 7.0}};

        xt::xarray<double> arr2
         {5.0, 6.0, 7.0};

        xt::xarray<double> res = xt::view(arr1, 1) + arr2;
        EXPECT_EQ(7., res(0));
        EXPECT_EQ(11., res(1));
        EXPECT_EQ(14., res(2));
    }

    TEST(xview, access)
    {
        xt::xarray<double> arr
        {{ 1.0, 2.0, 3.0 },
         { 2.0, 5.0, 7.0 },
         { 2.0, 5.0, 7.0 }};

        auto v1 = xt::view(arr, 1, xt::range(1, 3));
        EXPECT_EQ(v1(), arr(0, 1));
        EXPECT_EQ(v1(1), arr(1, 2));
        EXPECT_EQ(v1(1, 1), arr(1, 2));

        auto v2 = xt::view(arr, all(), newaxis(), all());
        //EXPECT_EQ(v2(1), arr(0, 1));
        EXPECT_EQ(v2(1, 0, 2), arr(1, 2));
        EXPECT_EQ(v2(2, 1, 0, 2), arr(1, 2));

        auto v3 = xt::view(arr, xt::range(0, 2), xt::range(1, 3));
        //EXPECT_EQ(v3(1), arr(0, 2));
        EXPECT_EQ(v3(1, 1), arr(1, 2));
        EXPECT_EQ(v3(2, 3, 1, 1), arr(1, 2));
    }

    TEST(xview, unchecked)
    {
        xt::xarray<double> arr
        { { 1.0, 2.0, 3.0 },
        { 2.0, 5.0, 7.0 },
        { 2.0, 5.0, 7.0 } };

        auto v1 = xt::view(arr, 1, xt::range(1, 3));
        EXPECT_EQ(v1.unchecked(1), arr(1, 2));

        auto v2 = xt::view(arr, all(), newaxis(), all());
        EXPECT_EQ(v2.unchecked(1, 0, 2), arr(1, 2));

        auto v3 = xt::view(arr, xt::range(0, 2), xt::range(1, 3));
        EXPECT_EQ(v3.unchecked(1, 1), arr(1, 2));
    }

    TEST(xview, iterator)
    {
        view_shape_type shape = {2, 3, 4};
        xarray<double, layout_type::row_major> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, range(0, 2), 1, range(1, 4));
        auto iter = view1.template begin<layout_type::row_major>();
        auto iter_end = view1.template end<layout_type::row_major>();

        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(18, *iter);
        ++iter;
        EXPECT_EQ(19, *iter);
        ++iter;
        EXPECT_EQ(20, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);

        auto view2 = view(view1, range(0, 2), range(1, 3));
        auto iter2 = view2.template begin<layout_type::row_major>();
        auto iter_end2 = view2.template end<layout_type::row_major>();

        EXPECT_EQ(7, *iter2);
        ++iter2;
        EXPECT_EQ(8, *iter2);
        ++iter2;
        EXPECT_EQ(19, *iter2);
        ++iter2;
        EXPECT_EQ(20, *iter2);
        ++iter2;
        EXPECT_EQ(iter2, iter_end2);
    }

    TEST(xview, fill)
    {
        view_shape_type shape = { 2, 3, 4 };
        xarray<double, layout_type::row_major> a(shape), res(shape);
        std::vector<double> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 };
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());
        std::vector<double> data_res = { 1, 2, 3, 4, 5, 4, 4, 4, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 4, 4, 4, 21, 22, 23, 24 };
        std::copy(data_res.cbegin(), data_res.cend(), res.template begin<layout_type::row_major>());
        auto view1 = view(a, range(0, 2), 1, range(1, 4));
        view1.fill(4);
        EXPECT_EQ(a, res);
    }

    TEST(xview, reverse_iterator)
    {
        view_shape_type shape = {2, 3, 4};
        xarray<double, layout_type::row_major> a(shape);
        std::vector<double> data = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, range(0, 2), 1, range(1, 4));
        auto iter = view1.template rbegin<layout_type::row_major>();
        auto iter_end = view1.template rend<layout_type::row_major>();

        EXPECT_EQ(20, *iter);
        ++iter;
        EXPECT_EQ(19, *iter);
        ++iter;
        EXPECT_EQ(18, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);

        auto view2 = view(view1, range(0, 2), range(1, 3));
        auto iter2 = view2.template rbegin<layout_type::row_major>();
        auto iter_end2 = view2.template rend<layout_type::row_major>();

        EXPECT_EQ(20, *iter2);
        ++iter2;
        EXPECT_EQ(19, *iter2);
        ++iter2;
        EXPECT_EQ(8, *iter2);
        ++iter2;
        EXPECT_EQ(7, *iter2);
        ++iter2;
        EXPECT_EQ(iter2, iter_end2);
    }

    TEST(xview, xview_on_xfunction)
    {
        view_shape_type shape = {3, 4};
        xarray<int> a(shape);
        std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        view_shape_type shape2 = {4};
        xarray<int> b(shape2);
        std::vector<int> data2 = {1, 2, 3, 4};
        std::copy(data2.cbegin(), data2.cend(), b.template begin<layout_type::row_major>());

        auto v = view(a + b, 1, range(1, 4));
        auto iter = v.begin();
        auto iter_end = v.end();

        EXPECT_EQ(8, *iter);
        ++iter;
        EXPECT_EQ(10, *iter);
        ++iter;
        EXPECT_EQ(12, *iter);
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xview, xview_on_xtensor)
    {
        xtensor<int, 2> a({3, 4});
        std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, 1, range(1, 4));
        EXPECT_EQ(a(1, 1), view1(0));
        EXPECT_EQ(a(1, 2), view1(1));
        EXPECT_EQ(size_t(1), view1.dimension());

        auto iter = view1.template begin<layout_type::row_major>();

        EXPECT_EQ(6, *iter);
        ++iter;
        EXPECT_EQ(7, *iter);
        ++iter;
        EXPECT_EQ(8, *iter);

        xarray<int> b({3}, 2);
        xtensor<int, 1> res = view1 + b;
        EXPECT_EQ(8, res(0));
        EXPECT_EQ(9, res(1));
        EXPECT_EQ(10, res(2));
    }

    TEST(xview, on_const_array)
    {
        const xt::xarray<int> a1{{0, 1}, {2, 3}};
        auto a2 = xt::view(a1, 1, xt::range(1, 2));
        int v2 = a2(0);
        EXPECT_EQ(v2, 3);

        auto it = a2.begin();
    }

    TEST(xview, trivial_iterating)
    {
        using tensor_type = xtensor<double, 1>;
        using shape_type = tensor_type::shape_type;
        tensor_type arr1{shape_type{2}};
        std::fill(arr1.begin(), arr1.end(), 6);
        auto view = xt::view(arr1, 0);
        auto iter = view.begin();
        auto iter_end = view.end();
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xview, const_trivial_iterating)
    {
        using tensor_type = xtensor<double, 1>;
        using shape_type = tensor_type::shape_type;
        tensor_type arr1{shape_type{2}};
        std::fill(arr1.begin(), arr1.end(), 6);
        const tensor_type arr2 = arr1;
        auto view = xt::view(arr2, 0);
        auto iter = view.begin();
        auto iter_end = view.end();
        ++iter;
        EXPECT_EQ(iter, iter_end);
    }

    TEST(xview, const_view)
    {
        typename xtensor<double, 3>::shape_type shape3 = {1, 2, 3};
        typename xtensor<double, 2>::shape_type shape2 = {2, 3};
        const xtensor<double, 3> arr(shape3, 2.5);
        xtensor<double, 2> arr2(shape2, 0.0);
        xtensor<double, 2> ref(shape2, 2.5);
        arr2 = xt::view(arr, 0);
        EXPECT_EQ(ref, arr2);
    }

    TEST(xview, newaxis_count)
    {
        size_t count1 = newaxis_count<xnewaxis<size_t>, xnewaxis<size_t>, xnewaxis<size_t>, xrange<size_t>>();
        EXPECT_EQ(count1, size_t(3));
        size_t count2 = newaxis_count<xnewaxis<size_t>, xrange<size_t>, xnewaxis<size_t>>();
        EXPECT_EQ(count2, size_t(2));
        size_t count3 = newaxis_count_before<xnewaxis<size_t>, xnewaxis<size_t>, xnewaxis<size_t>, xrange<size_t>>(3);
        EXPECT_EQ(count3, size_t(3));
        size_t count4 = newaxis_count_before<xnewaxis<size_t>, xrange<size_t>, xnewaxis<size_t>>(2);
        EXPECT_EQ(count4, size_t(1));
    }

    TEST(xview, newaxis)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, all(), newaxis(), all());
        EXPECT_EQ(a(1, 1), view1(1, 0, 1));
        EXPECT_EQ(a(1, 2), view1(1, 0, 2));
        EXPECT_EQ(size_t(3), view1.dimension());
        EXPECT_EQ(size_t(3), view1.shape()[0]);
        EXPECT_EQ(size_t(1), view1.shape()[1]);
        EXPECT_EQ(size_t(4), view1.shape()[2]);

        auto view2 = view(a, all(), all(), newaxis());
        EXPECT_EQ(a(1, 1), view2(1, 1, 0));
        EXPECT_EQ(a(1, 2), view2(1, 2, 0));
        EXPECT_EQ(size_t(3), view2.dimension());
        EXPECT_EQ(size_t(3), view2.shape()[0]);
        EXPECT_EQ(size_t(4), view2.shape()[1]);
        EXPECT_EQ(size_t(1), view2.shape()[2]);

        auto view3 = view(a, 1, newaxis(), all());
        EXPECT_EQ(a(1, 1), view3(0, 1));
        EXPECT_EQ(a(1, 2), view3(0, 2));
        EXPECT_EQ(size_t(2), view3.dimension());

        auto view4 = view(a, 1, all(), newaxis());
        EXPECT_EQ(a(1, 1), view4(1, 0));
        EXPECT_EQ(a(1, 2), view4(2, 0));
        EXPECT_EQ(size_t(2), view4.dimension());

        auto view5 = view(view1, 1);
        EXPECT_EQ(a(1, 1), view5(0, 1));
        EXPECT_EQ(a(1, 2), view5(0, 2));
        EXPECT_EQ(size_t(2), view5.dimension());

        auto view6 = view(view2, 1);
        EXPECT_EQ(a(1, 1), view6(1, 0));
        EXPECT_EQ(a(1, 2), view6(2, 0));
        EXPECT_EQ(size_t(2), view6.dimension());

        std::array<std::size_t, 3> idx1 = {1, 0, 2};
        EXPECT_EQ(a(1, 2), view1.element(idx1.begin(), idx1.end()));

        std::array<std::size_t, 3> idx2 = {1, 2, 0};
        EXPECT_EQ(a(1, 2), view2.element(idx2.begin(), idx2.end()));

        std::array<std::size_t, 3> idx3 = {1, 2};
        EXPECT_EQ(a(1, 2), view3.element(idx3.begin(), idx3.end()));
    }

    TEST(xview, newaxis_iterating)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        auto view1 = view(a, all(), all(), newaxis());
        auto iter1 = view1.template begin<layout_type::row_major>();
        auto iter1_end = view1.template end<layout_type::row_major>();

        EXPECT_EQ(a(0, 0), *iter1);
        ++iter1;
        EXPECT_EQ(a(0, 1), *iter1);
        ++iter1;
        EXPECT_EQ(a(0, 2), *iter1);
        ++iter1;
        EXPECT_EQ(a(0, 3), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 0), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 1), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 2), *iter1);
        ++iter1;
        EXPECT_EQ(a(1, 3), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 0), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 1), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 2), *iter1);
        ++iter1;
        EXPECT_EQ(a(2, 3), *iter1);
        ++iter1;
        EXPECT_EQ(iter1_end, iter1);

        auto view2 = view(a, all(), newaxis(), all());
        auto iter2 = view2.template begin<layout_type::row_major>();
        auto iter2_end = view2.template end<layout_type::row_major>();

        EXPECT_EQ(a(0, 0), *iter2);
        ++iter2;
        EXPECT_EQ(a(0, 1), *iter2);
        ++iter2;
        EXPECT_EQ(a(0, 2), *iter2);
        ++iter2;
        EXPECT_EQ(a(0, 3), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 0), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 1), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 2), *iter2);
        ++iter2;
        EXPECT_EQ(a(1, 3), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 0), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 1), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 2), *iter2);
        ++iter2;
        EXPECT_EQ(a(2, 3), *iter2);
        ++iter2;
        EXPECT_EQ(iter2_end, iter2);
    }

    TEST(xview, newaxis_function)
    {
        view_shape_type shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.cbegin(), data.cend(), a.template begin<layout_type::row_major>());

        xarray<double> b(view_shape_type(1, 4));
        std::copy(data.cbegin(), data.cbegin() + 4, b.template begin<layout_type::row_major>());

        auto v = view(b, newaxis(), all());
        xarray<double> res = a + v;

        std::vector<double> data2{2, 4, 6, 8, 6, 8, 10, 12, 10, 12, 14, 16};
        xarray<double> expected(shape);
        std::copy(data2.cbegin(), data2.cend(), expected.template begin<layout_type::row_major>());

        EXPECT_EQ(expected, res);
    }

    TEST(xview, range_adaptor)
    {
        using namespace xt::placeholders;
        using t = xarray<int>;
        t a = {1, 2, 3, 4, 5};

        auto n = xnone();

        auto v1 = view(a, range(3, _));
        t v1e = {4, 5};
        EXPECT_TRUE(v1e == v1);

        auto v2 = view(a, range(_, 2));
        t v2e = {1, 2};
        EXPECT_TRUE(v2e == v2);

        auto v3 = view(a, range(n, n));
        t v3e = {1, 2, 3, 4, 5};
        EXPECT_TRUE(v3e == v3);

        auto v4 = view(a, range(n, 2, -1));
        t v4e = {5, 4};
        EXPECT_TRUE(v4e == v4);

        auto v5 = view(a, range(2, n, -1));
        t v5e = {3, 2, 1};
        EXPECT_TRUE(v5e == v5);

        auto v6 = view(a, range(n, n, n));
        t v6e = {1, 2, 3, 4, 5};
        EXPECT_TRUE(v6e == v6);

        auto v7 = view(a, range(1, n, 2));
        t v7e = {2, 4};
        EXPECT_TRUE(v7e == v7);

        auto v8 = view(a, range(2, n, 2));
        t v8e = {3, 5};
        EXPECT_TRUE(v8e == v8);
    }

    TEST(xview, data_interface)
    {
        using namespace xt::placeholders;
        using T = xarray<int>;
        xarray<int> a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        using shape_type = typename T::shape_type;
        using index_type = typename T::shape_type;
        using size_type = typename T::size_type;

        auto next_idx = [](index_type& idx, const shape_type& shape)
        {
            for (size_type j = shape.size(); j != 0; --j)
            {
                size_type i = j - 1;
                if (idx[i] >= shape[i] - 1)
                {
                    idx[i] = 0;
                }
                else
                {
                    idx[i]++;
                    return idx;
                }
            }
            // return empty index, happens at last iteration step, but remains unused
            return index_type();
        };

        auto v1 = view(a, xt::all(), 1);
        auto shape1 = v1.shape();
        auto idx1 = index_type(shape1.size(), 0);
        auto strides1 = v1.strides();
        for (std::size_t i = 0; i < v1.size(); ++i)
        {
            auto linear_idx = std::inner_product(idx1.begin(), idx1.end(), strides1.begin(), std::size_t(0));
            EXPECT_EQ(v1[idx1], v1.data()[v1.data_offset() + linear_idx]);
            next_idx(idx1, shape1);
        }

        auto v2 = view(a, 1, range(_, _, 2));
        auto shape2 = v2.shape();
        auto idx2 = index_type(shape2.size(), 0);
        auto strides2 = v2.strides();
        for (std::size_t i = 0; i < v2.size(); ++i)
        {
            auto linear_idx = std::inner_product(idx2.begin(), idx2.end(), strides2.begin(), std::size_t(0));
            EXPECT_EQ(v2[idx2], v2.data()[v2.data_offset() + linear_idx]);
            next_idx(idx2, shape2);
        }
    }

    TEST(xview, strides_type)
    {
        xt::xtensor<float, 2> a{
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        };
        auto row = xt::view(a, 1, xt::all());
        bool cond1 = std::is_same<decltype(row)::strides_type, std::array<std::ptrdiff_t, 1>>::value;
        bool cond2 = std::is_same<decltype(row.strides()), const std::array<std::ptrdiff_t, 1>&>::value;
        EXPECT_TRUE(cond1);
        EXPECT_TRUE(cond2);
    }

    TEST(xview, transpose)
    {
        xt::xarray<int> vector = xt::linspace(1, 10, 10);
        auto matrix = xt::view(vector, xt::all(), xt::newaxis());
        auto mt = xt::transpose(matrix);
        EXPECT_EQ(mt.shape(), std::vector<std::size_t>({1, 10}));
        EXPECT_EQ(mt.strides(), std::vector<std::ptrdiff_t>({0, 1}));
        int sum = 0;
        for (std::size_t i = 0; i < vector.size(); ++i)
        {
            sum += mt(0, i);
        }
        EXPECT_EQ(55, sum);
    }

    TEST(xview, incompatible_shape)
    {
        xarray<int> a = xarray<int>::from_shape({4, 3, 2});
        xarray<int> b = xarray<int>::from_shape({2, 3, 4});
        auto v = view(a, all());

        EXPECT_FALSE(broadcastable(v.shape(), b.shape()));
        EXPECT_FALSE(broadcastable(b.shape(), v.shape()));
        EXPECT_THROW(assert_compatible_shape(b, v), broadcast_error);
        EXPECT_THROW(assert_compatible_shape(v, b), broadcast_error);
        EXPECT_THROW(v = b, broadcast_error);
        EXPECT_THROW(noalias(v) = b, broadcast_error);
    }

    TEST(xview, strides)
    {
        // Strides: 72/24/6/1
        xarray<int, layout_type::row_major> a = xarray<int, layout_type::row_major>::from_shape({5, 3, 4, 6});

        using strides_type = std::vector<std::ptrdiff_t>;
        auto s1 = view(a, 1, 1, xt::all(), xt::all()).strides();
        strides_type s1e = {6, 1};
        EXPECT_EQ(s1, s1e);

        auto s2 = view(a, 1, xt::all(), xt::all(), 1).strides();
        strides_type s2e = {24, 6};
        EXPECT_EQ(s2, s2e);

        auto s3 = view(a, 1, xt::all(), 1, xt::newaxis(), xt::newaxis(), xt::all()).strides();
        strides_type s3e = {24, 0, 0, 1};
        EXPECT_EQ(s3, s3e);

        auto s4 = view(a, xt::range(0, 1, 2), 1, 0, xt::all(), xt::newaxis()).strides();
        strides_type s4e = {0, 1, 0};
        EXPECT_EQ(s4, s4e);

        auto s4x = view(a, xt::range(0, 5, 2), 1, 0, xt::all(), xt::newaxis()).strides();
        strides_type s4xe = {72 * 2, 1, 0};
        EXPECT_EQ(s4x, s4xe);

        auto s5 = view(a, xt::all(), 1).strides();
        strides_type s5e = {72, 6, 1};
        EXPECT_EQ(s5, s5e);

        auto s6 = view(a, xt::all(), 1, 1, xt::newaxis(), xt::all()).strides();
        strides_type s6e = {72, 0, 1};
        EXPECT_EQ(s6, s6e);

        auto s7 = view(a, xt::all(), 1, xt::newaxis(), xt::all()).strides();
        strides_type s7e = {72, 0, 6, 1};
        EXPECT_EQ(s7, s7e);
    }

    TEST(xview, to_scalar)
    {
        std::array<std::size_t, 3> sh{2,2,2};
        xtensor<double, 3> a(sh);
        xtensor_fixed<double, xshape<2, 2, 2>> af = a;
        xarray<double> b = a;

        auto av = view(a, 1, 1);
        const auto av1 = view(a, 1, 1, 0);
        const double& ad1 = av1;
        EXPECT_EQ(ad1, av1());

        bool ax = is_xscalar<std::decay_t<decltype(av)>>::value;
        EXPECT_FALSE(ax);
        ax = is_xscalar<std::decay_t<decltype(av1)>>::value;
        EXPECT_TRUE(ax);
        auto bv = view(b, 1, 1, 1);
        ax = is_xscalar<decltype(bv)>::value;
        EXPECT_FALSE(ax);

        auto afv = view(af, 1, 1);
        auto afv1 = view(af, 1, 1, 0);

        double& afd1 = view(af, 1, 1, 0);
        EXPECT_EQ(afd1, af(1, 1, 0));
        ax = is_xscalar<decltype(afv)>::value;
        EXPECT_FALSE(ax);
        ax = is_xscalar<decltype(afv1)>::value;
        EXPECT_TRUE(ax);

        const xtensor<double, 2> ac = {{1,2}, {3,4}};
        double a1 = view(ac, 0, 0);
        const double& a2 = view(ac, 0, 0);

        EXPECT_EQ(a1, a2);

        double conv = av1;
        double conv1 = afv1;
        EXPECT_EQ(conv, conv1);
    }

    template <class V, class A>
    inline void test_view_iter(V& v, A& exp)
    {
        auto iter_expv1 = exp.begin();
        for (auto iter = v.begin(); iter != v.end(); ++iter)
        {
            EXPECT_EQ(*iter, *iter_expv1);
            ++iter_expv1;
        }

        auto riter_expv1 = exp.rbegin();
        for (auto iter = v.rbegin(); iter != v.rend(); ++iter)
        {
            EXPECT_EQ(*iter, *riter_expv1);
            ++riter_expv1;
        }

        auto rciter_expv1 = exp.template rbegin<layout_type::column_major>();
        for (auto iter = v.template rbegin<layout_type::column_major>();
             iter != v.template rend<layout_type::column_major>(); ++iter)
        {
            EXPECT_EQ(*iter, *rciter_expv1);
            ++rciter_expv1;
        }
    }

    TEST(xview, random_stepper)
    {
        xt::xarray<double> x = xt::arange(0, 100);
        x.reshape({5, 5, 4});

        xt::xarray<double> expected;
        if (XTENSOR_DEFAULT_LAYOUT == layout_type::row_major)
        {
           expected = {
               0, 1, 2, 3,
               20, 21, 22, 23,
               40, 41, 42, 43,
               60, 61, 62, 63,
               80, 81, 82, 83
           };
        }
        else
        {
           expected = {
               0, 1, 2, 3, 4,
               25, 26, 27, 28, 29,
               50, 51, 52, 53, 54,
               75, 76, 77, 78, 79
           };
        }
        auto v = xt::view(x, all(), 0);

        auto it1 = v.begin();
        auto it2 = v.storage_begin();
        auto it3 = v.rbegin();
        auto it4 = v.storage_rbegin();
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            std::ptrdiff_t ix = static_cast<std::ptrdiff_t>(i);
            EXPECT_EQ(*(it1 + ix), expected[i]);
            EXPECT_EQ(*(it2 + ix), expected[i]);
            EXPECT_EQ(*(it3 + ix), expected[expected.size() - 1 - i]);
            EXPECT_EQ(*(it4 + ix), expected[expected.size() - 1 - i]);
        }
    }

    TEST(xview, keep_slice)
    {
        xtensor<double, 3, layout_type::row_major> a = {{{ 1, 2, 3, 4},
                                                         { 5, 6, 7, 8}},
                                                        {{ 9,10,11,12},
                                                         {13,14,15,16}},
                                                        {{17,18,19,20},
                                                         {21,22,23,24}}};

        auto v1 = xt::view(a, keep(1), keep(0, 1), keep(0, 3));
        xtensor<double, 3> exp_v1 = {{{9, 12}, {13, 16}}};
        EXPECT_EQ(v1, exp_v1);

        test_view_iter(v1, exp_v1);

        auto v2 = xt::view(a, keep(1), xt::all(), xt::range(0, xt::xnone(), 3));
        EXPECT_EQ(v2, v1);
        EXPECT_EQ(v2, exp_v1);

        auto v3 = xt::view(a, keep(1), keep(1, 1, 1, 1), keep(0, 3));
        xtensor<double, 3> exp_v3 = {{{13, 16}, {13, 16}, {13, 16}, {13, 16}}};
        EXPECT_EQ(v3, exp_v3);

        test_view_iter(v3, exp_v3);

        auto v4 = xt::view(a, keep(0, 2), keep(0));
        xtensor<double, 3> exp_v4 = {{{  1.,   2.,   3.,   4.}},
                                     {{ 17.,  18.,  19.,  20.}}};
        EXPECT_EQ(v4, exp_v4);

        v4(0, 0) = 123;
        v4(1, 0) = 123;
        EXPECT_EQ(a(0, 0, 0), 123);
        EXPECT_EQ(a(1, 0, 0), 123);

        v3(0, 2, 1) = 1000;
        EXPECT_EQ(a(1, 1, 3), 1000);

        bool b = detail::slices_contigous<xkeep_slice<int>, int>::value;
        EXPECT_FALSE(b);
        b = detail::slices_contigous<xrange<int>, xrange<int>, int>::value;
        EXPECT_TRUE(b);
    }

    TEST(xview, keep_negative)
    {
        xtensor<double, 3, layout_type::row_major> a = {{{ 1, 2, 3, 4},
                                                         { 5, 6, 7, 8}},
                                                        {{ 9,10,11,12},
                                                         {13,14,15,16}},
                                                        {{17,18,19,20},
                                                         {21,22,23,24}}};

        auto v1 = xt::view(a, keep(-2), keep(-0, -1), keep(0, -1));
        xtensor<double, 3> exp_v1 = {{{9, 12}, {13, 16}}};
        EXPECT_EQ(v1, exp_v1);

        // check compilation
        // std::vector<std::ptrdiff_t> ixx({1,2});
        // xt::xview v2(a, keep(ixx));
        // EXPECT_EQ(xt::xview(a, keep({1, 2})), v2);
    }

    TEST(xview, mixed_types)
    {
        xt::xarray<std::uint8_t> input;
        xt::xarray<float> output;
        input.resize({ { 50,16,16,3 } });
        output.resize({ { 50,16,16,3 } });

        input.fill(std::uint8_t(1));
        output.fill(float(2.));
        for (int i = 0; i<50; ++i)
        {
            auto in_view = xt::view(input, i);
            auto out_view = xt::view(output, i);
            out_view = in_view;
        }

        EXPECT_EQ(output(0, 5, 5, 2), 1);
    }

    TEST(xview, where_operation)
    {
        xt::xtensor<size_t, 2> I = {{0, 0}, {1, 1}, {2, 2}};
        auto col = xt::view(I, xt::all(), 0);
        auto idx = xt::where(xt::equal(col, size_t(0)));

        std::array<std::size_t, 1> exp_idx = {0};
        EXPECT_EQ(idx[0], exp_idx);

        auto idx2 = xt::where(col > size_t(0));
        EXPECT_EQ(idx2.size(), 2);
        exp_idx[0] = 1;
        EXPECT_EQ(idx2[0], exp_idx);
        exp_idx[0] = 2;
        EXPECT_EQ(idx2[1], exp_idx);
    }
}