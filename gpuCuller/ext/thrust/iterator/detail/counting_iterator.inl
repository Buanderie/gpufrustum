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

#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/numeric_traits.h>

namespace thrust
{

// forward declaration of counting_iterator
template <typename Incrementable, typename Space, typename Traversal, typename Difference>
  class counting_iterator;

namespace detail
{

template <typename Incrementable, typename Space, typename Traversal, typename Difference>
  struct counting_iterator_base
{
  typedef typename thrust::experimental::detail::ia_dflt_help<
    Space,
    thrust::detail::identity_<thrust::any_space_tag>
  >::type space;

  typedef typename thrust::experimental::detail::ia_dflt_help<
      Traversal,
      thrust::detail::eval_if<
          thrust::detail::is_numeric<Incrementable>::value,
          thrust::detail::identity_<random_access_traversal_tag>,
          thrust::iterator_traversal<Incrementable>
      >
  >::type traversal;

  // XXX this is equivalent to Boost's implementation
  //typedef typename detail::ia_dflt_help<
  //  Difference,
  //  eval_if<
  //    is_numeric<Incrementable>::value,
  //    numeric_difference<Incrementable>,
  //    iterator_difference<Incrementable>
  //  >
  //>::type difference;

  typedef typename thrust::experimental::detail::ia_dflt_help<
    Difference,
    thrust::detail::eval_if<
      thrust::detail::is_numeric<Incrementable>::value,
      thrust::detail::identity_<std::ptrdiff_t>,
      thrust::iterator_difference<Incrementable>
    >
  >::type difference;

  typedef thrust::experimental::iterator_adaptor<
    counting_iterator<Incrementable, Space, Traversal, Difference>, // self
    Incrementable,                                                  // Base
    Incrementable *,                                                // Pointer -- maybe we should make this device_ptr when memory space category is device?
    Incrementable,                                                  // Value
    space,
    traversal,
    Incrementable const &,
    difference
  > type;
}; // end counting_iterator_base


// specialize iterator_device_reference for counting_iterator
// transform_iterator returns the same reference on the device as on the host
template <typename Incrementable, typename Space, typename Traversal, typename Difference>
  struct iterator_device_reference<
    thrust::counting_iterator<
      Incrementable, Space, Traversal, Difference
    >
  >
{
  typedef typename thrust::iterator_traits< thrust::counting_iterator<Incrementable,Space,Traversal,Difference> >::reference type;
}; // end iterator_device_reference


namespace device
{

template<typename Incrementable, typename Space, typename Traversal, typename Difference>
  inline __device__
    typename iterator_device_reference< thrust::counting_iterator<Incrementable,Space,Traversal,Difference> >::type
      dereference(thrust::counting_iterator<Incrementable,Space,Traversal,Difference> iter)
{
  return *iter;
} // end dereference()

template<typename Incrementable, typename Space, typename Traversal, typename Difference, typename IndexType>
  inline __device__
    typename iterator_device_reference< thrust::counting_iterator<Incrementable,Space,Traversal,Difference> >::type
      dereference(thrust::counting_iterator<Incrementable,Space,Traversal,Difference> iter, IndexType n)
{
  return iter[n];
} // end dereference()

} // end device


template<typename Difference, typename Incrementable1, typename Incrementable2>
  struct iterator_distance
{
  __host__ __device__
  static Difference distance(Incrementable1 x, Incrementable2 y)
  {
    return y - x;
  }
};


template<typename Difference, typename Incrementable1, typename Incrementable2>
  struct number_distance
{
  __host__ __device__
  static Difference distance(Incrementable1 x, Incrementable2 y)
  {
    return numeric_distance(x,y);
  }
};

} // end detail

} // end thrust

