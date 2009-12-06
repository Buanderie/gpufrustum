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


/*! \file device_reference.inl
 *  \brief Inline file for device_reference.h.
 */

#include <thrust/device_reference.h>
#include <thrust/copy.h>
#include <thrust/detail/type_traits.h>

namespace thrust
{

template<typename T>
  template<typename OtherT>
    device_reference<T>
      ::device_reference(const device_reference<OtherT> &ref

// XXX msvc screws this up
#ifndef _MSC_VER
                         , typename
                         detail::enable_if<
                           detail::is_convertible<
                             typename device_reference<OtherT>::pointer,
                             pointer
                           >::value
                         >::type *dummy
#endif // _MSC_VER
                        )
        :mPtr(ref.mPtr)
{
  ;
} // end device_reference::device_reference()

template<typename T>
  device_reference<T>
    ::device_reference(const pointer &ptr)
      :mPtr(ptr)
{
  ;
} // end device_reference::device_reference()

template<typename T>
  typename device_reference<T>::pointer device_reference<T>
    ::operator&(void) const
{
  return mPtr;
} // end device_reference::operator&()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator=(const T &v)
{
  thrust::copy(&v, &v + 1, mPtr);
  return *this;
} // end device_reference::operator=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator=(const device_reference &ref)
{
  thrust::copy(&ref, &ref + 1, mPtr);
  return *this;
} // end device_reference::operator=()

template<typename T>
  template<typename OtherT>
    device_reference<T> &device_reference<T>
      ::operator=(const device_reference<OtherT> &ref)
{
  thrust::copy(&ref, &ref + 1, mPtr);
  return *this;
} // end device_reference::operator=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator++(void)
{
  value_type temp = *this;
  ++temp;
  *this = temp;
  return *this;
} // end device_reference::operator++()

template<typename T>
  typename device_reference<T>::value_type
    device_reference<T>
      ::operator++(int)
{
  value_type temp = *this;
  value_type result = temp++;
  *this = temp;
  return result;
} // end device_reference::operator++()

template<typename T>
  device_reference<T>
    ::operator typename device_reference<T>::value_type (void) const
{
  value_type result;
  thrust::copy(mPtr, mPtr + 1, &result);
  return result;
} // end device_reference::operator value_type ()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator+=(const T &rhs)
{
  value_type temp = *this;
  temp += rhs;
  *this = temp;
  return *this;
} // end device_reference::operator+=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator--(void)
{
  value_type temp = *this;
  --temp;
  *this = temp;
  return *this;
} // end device_reference::operator--()

template<typename T>
  typename device_reference<T>::value_type
    device_reference<T>
      ::operator--(int)
{
  value_type temp = *this;
  value_type result = temp--;
  *this = temp;
  return result;
} // end device_reference::operator--()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator-=(const T &rhs)
{
  value_type temp = *this;
  temp -= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator-=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator*=(const T &rhs)
{
  value_type temp = *this;
  temp *= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator*=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator/=(const T &rhs)
{
  value_type temp = *this;
  temp /= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator/=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator%=(const T &rhs)
{
  value_type temp = *this;
  temp %= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator%=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator<<=(const T &rhs)
{
  value_type temp = *this;
  temp <<= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator<<=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator>>=(const T &rhs)
{
  value_type temp = *this;
  temp >>= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator>>=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator&=(const T &rhs)
{
  value_type temp = *this;
  temp &= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator&=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator|=(const T &rhs)
{
  value_type temp = *this;
  temp |= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator|=()

template<typename T>
  device_reference<T> &device_reference<T>
    ::operator^=(const T &rhs)
{
  value_type temp = *this;
  temp ^= rhs;
  *this = temp;
  return *this;
} // end device_reference::operator^=()

} // end thrust

