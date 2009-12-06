/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#include <algorithm>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/raw_buffer.h>

/*
 * The merge() function is derived from the function of the same name
 * in stl_algo.h of the SGI STL implementation. The orginial license 
 * follows below.
 *
 * http://www.sgi.com/tech/stl/stl_algo.h
 */

/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 *
 * Copyright (c) 1996
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */


namespace thrust
{

namespace sorting
{

namespace detail
{

namespace host
{

template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void merge_by_key(RandomAccessIterator1 keys_begin,
                    RandomAccessIterator1 keys_middle,
                    RandomAccessIterator1 keys_end,
                    RandomAccessIterator2 values_begin,
                    StrictWeakOrdering comp,
                    size_t len1, size_t len2)
{
    if(len1 == 0 || len2 == 0)
        return;

    RandomAccessIterator2 values_middle = values_begin + len1;

    if(len1 + len2 == 2)
    {
        if(comp(*keys_middle, *keys_begin))
        {
            std::iter_swap(keys_begin, keys_middle);
            std::iter_swap(values_begin, values_middle);
        } // end if

        return;
    } // end if

    RandomAccessIterator1 keys_first_cut = keys_begin;
    RandomAccessIterator1 keys_second_cut = keys_middle;
    RandomAccessIterator2 values_first_cut = values_begin;
    RandomAccessIterator2 values_second_cut = values_middle;

    typename thrust::iterator_traits<RandomAccessIterator1>::difference_type len11 = 0;
    typename thrust::iterator_traits<RandomAccessIterator1>::difference_type len22 = 0;

    if(len1 > len2)
    {
        len11 = len1 / 2;
        std::advance(keys_first_cut, len11);
        std::advance(values_first_cut, len11);
        keys_second_cut = std::lower_bound(keys_middle, keys_end, *keys_first_cut, comp);
        len22 = std::distance(keys_middle, keys_second_cut);

        std::advance(values_second_cut, len22);
    } // end if
    else
    {
        len22 = len2 / 2;
        std::advance(keys_second_cut, len22);
        std::advance(values_second_cut, len22);
        keys_first_cut = std::upper_bound(keys_begin, keys_middle, *keys_second_cut, comp);
        len11 = std::distance(keys_begin, keys_first_cut);

        std::advance(values_first_cut, len11);
    } // end else

    std::rotate(keys_first_cut, keys_middle, keys_second_cut);
    std::rotate(values_first_cut, values_middle, values_second_cut);

    RandomAccessIterator1 new_keys_middle = keys_first_cut;
    std::advance(new_keys_middle, std::distance(keys_middle, keys_second_cut));

    RandomAccessIterator2 new_values_middle = values_first_cut;
    std::advance(new_values_middle, std::distance(values_middle, values_second_cut));

    merge_by_key(keys_begin, keys_first_cut, new_keys_middle, values_begin,
            comp, len11, len22);
    merge_by_key(new_keys_middle, keys_second_cut, keys_end, new_values_middle,
            comp, len1 - len11, len2 - len22);
} // end merge_by_key()


// \see http://thomas.baudel.name/Visualisation/VisuTri/inplacestablesort.html
template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                                RandomAccessIterator1 keys_end,
                                RandomAccessIterator2 values_begin,
                                StrictWeakOrdering comp,
                                thrust::detail::true_type,
                                thrust::detail::true_type)
{
    typedef typename thrust::iterator_traits<RandomAccessIterator1>::value_type   KeyType;
    typedef typename thrust::iterator_traits<RandomAccessIterator2>::value_type ValueType;

    if(keys_end - keys_begin < 2) return;

    RandomAccessIterator1   keys_middle   = keys_begin   + (keys_end - keys_begin)/2;
    RandomAccessIterator2 values_middle = values_begin + (keys_end - keys_begin)/2;

    // sort each side
    thrust::sorting::stable_merge_sort_by_key(keys_begin, keys_middle, values_begin, comp);
    thrust::sorting::stable_merge_sort_by_key(keys_middle, keys_end, values_middle, comp);

    // merge
    merge_by_key(keys_begin, keys_middle, keys_end, values_begin,
            comp, keys_middle - keys_begin, keys_end - keys_middle);
} // end stable_merge_sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                                RandomAccessIterator1 keys_end,
                                RandomAccessIterator2 values_begin,
                                StrictWeakOrdering comp,
                                thrust::detail::false_type,
                                thrust::detail::false_type)
{
  typedef typename iterator_value<RandomAccessIterator1>::type KeyType;
  typedef typename iterator_value<RandomAccessIterator2>::type ValueType;

  // copy input to temporary ranges
  thrust::detail::raw_host_buffer<KeyType>   keys_temp(keys_begin, keys_end);
  thrust::detail::raw_host_buffer<ValueType> values_temp(values_begin, values_begin + keys_end - keys_begin);

  stable_merge_sort_by_key(keys_temp.begin(), keys_temp.end(),
                           values_begin.begin(),
                           comp,
                           thrust::detail::true_type(),
                           thrust::detail::true_type());

  // copy to original input
  thrust::copy(keys_temp.begin(), keys_temp.end(), keys_begin);
  thrust::copy(values_temp.begin(), values_temp.end(), values_begin);
} // end stable_merge_sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                                RandomAccessIterator1 keys_end,
                                RandomAccessIterator2 values_begin,
                                StrictWeakOrdering comp,
                                thrust::detail::false_type,
                                thrust::detail::true_type)
{
  typedef typename iterator_value<RandomAccessIterator1>::type KeyType;

  // copy input to temporary ranges
  thrust::detail::raw_host_buffer<KeyType> keys_temp(keys_begin, keys_end);

  stable_merge_sort_by_key(keys_temp.begin(), keys_temp.end(),
                           values_begin,
                           comp,
                           thrust::detail::true_type(),
                           thrust::detail::true_type());

  // copy to original input
  thrust::copy(keys_temp.begin(), keys_temp.end(), keys_begin);
} // end stable_merge_sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                                RandomAccessIterator1 keys_end,
                                RandomAccessIterator2 values_begin,
                                StrictWeakOrdering comp,
                                thrust::detail::true_type,
                                thrust::detail::false_type)
{
  typedef typename iterator_value<RandomAccessIterator2>::type ValueType;

  // copy input to temporary ranges
  RandomAccessIterator2 values_end = values_begin + (keys_end + keys_begin);
  thrust::detail::raw_host_buffer<ValueType> values_temp(values_begin, values_end);

  stable_merge_sort_by_key(keys_begin, keys_end,
                           values_temp.begin(),
                           comp,
                           thrust::detail::true_type(),
                           thrust::detail::true_type());

  // copy to original input
  thrust::copy(values_temp.begin(), values_temp.end(), values_begin);
} // end stable_merge_sort_by_key()


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_merge_sort_by_key(RandomAccessIterator1 keys_begin,
                                RandomAccessIterator1 keys_end,
                                RandomAccessIterator2 values_begin,
                                StrictWeakOrdering comp)
{
  // dispatch on whether or not the iterators are trivial
  return stable_merge_sort_by_key(keys_begin, keys_end, values_begin, comp,
          typename thrust::detail::is_trivial_iterator<RandomAccessIterator1>::type(),
          typename thrust::detail::is_trivial_iterator<RandomAccessIterator2>::type());
} // end stable_merge_sort_by_key()


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void merge(RandomAccessIterator begin,
             RandomAccessIterator middle,
             RandomAccessIterator end,
             StrictWeakOrdering comp,
             size_t len1, size_t len2)
{
    if(len1 == 0 || len2 == 0)
        return;

    if(len1 + len2 == 2)
    {
        if(comp(*middle, *begin))
        {
            std::iter_swap(begin, middle);
        } // end if

        return;
    } // end if

    RandomAccessIterator first_cut = begin;
    RandomAccessIterator second_cut = middle;

    typename thrust::iterator_traits<RandomAccessIterator>::difference_type len11 = 0;
    typename thrust::iterator_traits<RandomAccessIterator>::difference_type len22 = 0;
    if(len1 > len2)
    {
        len11 = len1 / 2;
        std::advance(first_cut, len11);
        second_cut = std::lower_bound(middle, end, *first_cut, comp);
        len22 = std::distance(middle, second_cut);
    } // end if
    else
    {
        len22 = len2 / 2;
        std::advance(second_cut, len22);
        first_cut = std::upper_bound(begin, middle, *second_cut, comp);
        len11 = std::distance(begin, first_cut);
    } // end else

    std::rotate(first_cut, middle, second_cut);

    RandomAccessIterator new_middle = first_cut;
    std::advance(new_middle, std::distance(middle, second_cut));

    merge(begin, first_cut, new_middle,
            comp, len11, len22);
    merge(new_middle, second_cut, end,
            comp, len1 - len11, len2 - len22);
} // end merge()


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_merge_sort(RandomAccessIterator begin,
                         RandomAccessIterator end,
                         StrictWeakOrdering comp,
                         thrust::detail::true_type)
{
    if(end - begin < 2) return;

    RandomAccessIterator middle = begin + (end - begin)/2;

    // sort each side
    stable_merge_sort(begin, middle, comp, thrust::detail::true_type());
    stable_merge_sort(middle, end, comp, thrust::detail::true_type());

    // merge
    merge(begin, middle, end,
            comp, middle - begin, end - middle);
} // end stable_merge_sort()

template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_merge_sort(RandomAccessIterator begin,
                         RandomAccessIterator end,
                         StrictWeakOrdering comp,
                         thrust::detail::false_type)
{
  typedef typename iterator_value<RandomAccessIterator>::type ValueType;

  // copy input to temporary range
  thrust::detail::raw_host_buffer<ValueType> temp(begin,end);

  stable_merge_sort(temp.begin(), temp.end(), comp, thrust::detail::true_type());

  // copy to original range
  thrust::copy(temp.begin(), temp.end(), begin);
} // end stable_merge_sort()


template<typename RandomAccessIterator,
         typename StrictWeakOrdering>
  void stable_merge_sort(RandomAccessIterator begin,
                         RandomAccessIterator end,
                         StrictWeakOrdering comp)
{
  // dispatch on whether or not the iterator is trivial
  return stable_merge_sort(begin, end, comp,
          typename thrust::detail::is_trivial_iterator<RandomAccessIterator>::type());
} // end stable_merge_sort()

} // end namespace host

} // end namespace detail

} // end namespace sorting

} // end namespace thrust

