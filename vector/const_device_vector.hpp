// Copyright (C) 2021 Attila Krasznahorkay.
#ifndef DETRAY_DATA_MODEL_CONST_DEVICE_VECTOR_HPP
#define DETRAY_DATA_MODEL_CONST_DEVICE_VECTOR_HPP

// Local include(s).
#include "core/types.hpp"

// System include(s).
#include <cassert>
#include <vector>

namespace detray::cuda {

   /// Class mimicking @c std::vector in CUDA device code
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< typename TYPE >
   class const_device_vector {

   public:
      /// @name Type definitions mimicking @c std::vector
      /// @{
      typedef std::size_t    size_type;
      typedef std::ptrdiff_t difference_type;
      typedef const TYPE*    const_pointer;
      typedef const TYPE&    const_reference;
      typedef TYPE           value_type;
      typedef typename std::vector< TYPE >::const_iterator   const_iterator;
      typedef typename std::vector< TYPE >::const_reverse_iterator
         const_reverse_iterator;
      /// @}

      /// Constructor, on top of a previously allocated/filled block of memory
      DETRAY_HOST_AND_DEVICE
      const_device_vector( size_type size, const_pointer ptr )
      : m_size( size ), m_ptr( ptr ) {}

      /// @name Vector element access functions
      /// @{
      DETRAY_HOST_AND_DEVICE
      const_reference at( size_type pos ) const {
         assert( pos < m_size );
         return m_ptr[ pos ];
      }

      DETRAY_HOST_AND_DEVICE
      const_reference operator[]( size_type pos ) const {
         return m_ptr[ pos ];
      }

      DETRAY_HOST_AND_DEVICE
      const_reference front() const {
         return m_ptr[ 0 ];
      }

      DETRAY_HOST_AND_DEVICE
      const_reference back() const {
         return m_ptr[ m_size - 1 ];
      }
      /// @}

      /// @name Iterator providing functions
      /// @{
      DETRAY_HOST
      const_iterator begin() const {
         return const_iterator( m_ptr );
      }
      DETRAY_HOST
      const_iterator cbegin() const {
         return begin();
      }

      DETRAY_HOST
      const_iterator end() const {
         return const_iterator( m_ptr + m_size );
      }
      DETRAY_HOST
      const_iterator cend() const {
         return const_iterator( m_ptr + m_size );
      }

      DETRAY_HOST
      const_reverse_iterator rbegin() const {
         return const_reverse_iterator( end() );
      }
      DETRAY_HOST
      const_reverse_iterator crbegin() const {
         return const_reverse_iterator( cend() );
      }

      DETRAY_HOST
      const_reverse_iterator rend() const {
         return const_reverse_iterator( begin() );
      }
      DETRAY_HOST
      const_reverse_iterator crend() const {
         return const_reverse_iterator( cbegin() );
      }
      /// @}

      /// @name Additional helper functions
      /// @{
      DETRAY_HOST_AND_DEVICE
      bool empty() const {
         return m_size == 0;
      }

      DETRAY_HOST_AND_DEVICE
      size_type size() const {
         return m_size;
      }
      /// @}

   private:
      /// Size of the array that this object looks at
      size_type m_size;
      /// Pointer to the start of the array
      const_pointer m_ptr;

   }; // class const_device_vector

} // namespace detray::cuda

#endif // DETRAY_DATA_MODEL_CONST_DEVICE_VECTOR_HPP
