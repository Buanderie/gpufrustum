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


/*! \file remove.inl
 *  \brief Inline file for remove.h
 */

#pragma once

#include <thrust/iterator/iterator_traits.h>

#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/detail/raw_buffer.h>

namespace thrust
{

namespace detail
{

namespace device
{


template<typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(ForwardIterator begin,
                            ForwardIterator end,
                            InputIterator stencil,
                            Predicate pred)
{
  // XXX do we need to call destructors for elements which get removed?

  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;

  // create temporary storage for an intermediate result
  typedef raw_device_buffer<InputType> RawBuffer;
  RawBuffer temp(end - begin);

  // remove into temp
  typename RawBuffer::iterator new_end = thrust::detail::device::remove_copy_if(begin, end, stencil, temp.begin(), pred);

  // copy temp to the original range
  thrust::copy(temp.begin(), new_end, begin);

  return begin + (new_end - temp.begin());
} 

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred)
{
    return thrust::detail::device::remove_copy_if(first, last, first, result, pred);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(InputIterator1 begin,
                                InputIterator1 end,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred)
{
  typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;

  difference_type n = end - begin;

  difference_type size_of_new_sequence = 0;
  if(n > 0)
  {
    // negate the predicate -- this tells us which elements to keep
    thrust::unary_negate<Predicate> not_pred(pred);

    // evaluate not_pred on [begin,end), store result to temp vector
    raw_device_buffer<difference_type> result_of_not_pred(n);

    thrust::transform(stencil,
                      stencil + n,
                      result_of_not_pred.begin(),
                      not_pred);

    // scan the pred result to a temp vector
    raw_device_buffer<difference_type> not_pred_scatter_indices(n);
    thrust::exclusive_scan(result_of_not_pred.begin(),
                           result_of_not_pred.end(),
                           not_pred_scatter_indices.begin());

    // scatter the true partition
    thrust::scatter_if(begin,
                       end,
                       not_pred_scatter_indices.begin(),
                       result_of_not_pred.begin(),
                       result);

    // find the end of the new sequence
    size_of_new_sequence = not_pred_scatter_indices[n - 1]
                         + (result_of_not_pred[n - 1] ? 1 : 0);
  } // end if

  return result + size_of_new_sequence;
} // end remove_copy_if()


} // end namespace device

} // end namespace detail

} // end namespace thrust


