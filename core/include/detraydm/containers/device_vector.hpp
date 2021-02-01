/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "detraydm/utils/cuda_types.hpp"

// System include(s).
#include <cassert>
#include <vector>

namespace detraydm::cuda {

   /// Class mimicking @c std::vector in CUDA device code
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< typename TYPE >
   class device_vector {

   public:
      /// @name Type definitions mimicking @c std::vector
      /// @{
      typedef std::size_t    size_type;
      typedef std::ptrdiff_t difference_type;
      typedef TYPE*          pointer;
      typedef const TYPE*    const_pointer;
      typedef TYPE&          reference;
      typedef const TYPE&    const_reference;
      typedef TYPE           value_type;
      typedef typename std::vector< TYPE >::iterator         iterator;
      typedef typename std::vector< TYPE >::const_iterator   const_iterator;
      typedef typename std::vector< TYPE >::reverse_iterator reverse_iterator;
      typedef typename std::vector< TYPE >::const_reverse_iterator
         const_reverse_iterator;
      /// @}

      /// Constructor, on top of a previously allocated/filled block of memory
      DETRAYDM_CUDA_HOST_AND_DEVICE
      device_vector( size_type size, pointer ptr )
      : m_size( size ), m_ptr( ptr ) {}

      /// @name Vector element access functions
      /// @{
      DETRAYDM_CUDA_HOST_AND_DEVICE
      reference at( size_type pos ) {
         assert( pos < m_size );
         return m_ptr[ pos ];
      }

      DETRAYDM_CUDA_HOST_AND_DEVICE
      const_reference at( size_type pos ) const {
         assert( pos < m_size );
         return m_ptr[ pos ];
      }

      DETRAYDM_CUDA_HOST_AND_DEVICE
      reference operator[]( size_type pos ) {
         return m_ptr[ pos ];
      }
      DETRAYDM_CUDA_HOST_AND_DEVICE
      const_reference operator[]( size_type pos ) const {
         return m_ptr[ pos ];
      }

      DETRAYDM_CUDA_HOST_AND_DEVICE
      reference front() {
         return m_ptr[ 0 ];
      }
      DETRAYDM_CUDA_HOST_AND_DEVICE
      const_reference front() const {
         return m_ptr[ 0 ];
      }

      DETRAYDM_CUDA_HOST_AND_DEVICE
      reference back() {
         return m_ptr[ m_size - 1 ];
      }
      DETRAYDM_CUDA_HOST_AND_DEVICE
      const_reference back() const {
         return m_ptr[ m_size - 1 ];
      }
      /// @}

      /// @name Iterator providing functions
      /// @{
      DETRAYDM_CUDA_HOST
      iterator begin() {
         return iterator( m_ptr );
      }
      DETRAYDM_CUDA_HOST
      const_iterator begin() const {
         return const_iterator( m_ptr );
      }
      DETRAYDM_CUDA_HOST
      const_iterator cbegin() const {
         return begin();
      }

      DETRAYDM_CUDA_HOST
      iterator end() {
         return iterator( m_ptr + m_size );
      }
      DETRAYDM_CUDA_HOST
      const_iterator end() const {
         return const_iterator( m_ptr + m_size );
      }
      DETRAYDM_CUDA_HOST
      const_iterator cend() const {
         return const_iterator( m_ptr + m_size );
      }

      DETRAYDM_CUDA_HOST
      reverse_iterator rbegin() {
         return reverse_iterator( end() );
      }
      DETRAYDM_CUDA_HOST
      const_reverse_iterator rbegin() const {
         return const_reverse_iterator( end() );
      }
      DETRAYDM_CUDA_HOST
      const_reverse_iterator crbegin() const {
         return const_reverse_iterator( cend() );
      }

      DETRAYDM_CUDA_HOST
      reverse_iterator rend() {
         return reverse_iterator( begin() );
      }
      DETRAYDM_CUDA_HOST
      const_reverse_iterator rend() const {
         return const_reverse_iterator( begin() );
      }
      DETRAYDM_CUDA_HOST
      const_reverse_iterator crend() const {
         return const_reverse_iterator( cbegin() );
      }
      /// @}

      /// @name Additional helper functions
      /// @{
      DETRAYDM_CUDA_HOST_AND_DEVICE
      bool empty() const {
         return m_size == 0;
      }

      DETRAYDM_CUDA_HOST_AND_DEVICE
      size_type size() const {
         return m_size;
      }
      /// @}

   private:
      /// Size of the array that this object looks at
      size_type m_size;
      /// Pointer to the start of the array
      pointer m_ptr;

   }; // class device_vector

} // namespace detraydm::cuda
