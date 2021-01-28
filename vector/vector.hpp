// Copyright (C) 2021 Attila Krasznahorkay. All rights reserved.
#ifndef DETRAY_DATA_MODEL_VECTOR_HPP
#define DETRAY_DATA_MODEL_VECTOR_HPP

// System include(s).
#include <stdexcept>
#include <vector>

namespace detray::cuda {

   /// Class mimicking @c std::vector in CUDA device code
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< typename TYPE >
   class vector {

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
      vector( size_type size, pointer ptr )
      : m_size( size ), m_ptr( ptr ) {}

      /// @name Vector element access functions
      /// @{
      reference at( size_type pos ) {
         if( pos >= m_size ) {
            throw std::out_of_range( "Invalid vector element requested" );
         }
         return m_ptr[ pos ];
      }

      const_reference at( size_type pos ) const {
         if( pos >= m_size ) {
            throw std::out_of_range();
         }
         return m_ptr[ pos ];
      }

      reference operator[]( size_type pos ) {
         return m_ptr[ pos ];
      }
      const_reference operator[]( size_type pos ) const {
         return m_ptr[ pos ];
      }

      reference front() {
         return m_ptr[ 0 ];
      }
      const_reference front() const {
         return m_ptr[ 0 ];
      }

      reference back() {
         return m_ptr[ m_size - 1 ];
      }
      const_reference back() const {
         return m_ptr[ m_size - 1 ];
      }
      /// @}

      /// @name Iterator providing functions
      /// @{
      iterator begin() {
         return iterator( m_ptr );
      }
      const_iterator begin() const {
         return const_iterator( m_ptr );
      }
      const_iterator cbegin() const {
         return begin();
      }

      iterator end() {
         return iterator( m_ptr + m_size );
      }
      const_iterator end() const {
         return const_iterator( m_ptr + m_size );
      }
      const_iterator cend() const {
         return const_iterator( m_ptr + m_size );
      }

      reverse_iterator rbegin() {
         return reverse_iterator( end() );
      }
      const_reverse_iterator rbegin() const {
         return const_reverse_iterator( end() );
      }
      const_reverse_iterator crbegin() const {
         return const_reverse_iterator( cend() );
      }

      reverse_iterator rend() {
         return reverse_iterator( begin() );
      }
      const_reverse_iterator rend() const {
         return const_reverse_iterator( begin() );
      }
      const_reverse_iterator crend() const {
         return const_reverse_iterator( cbegin() );
      }
      /// @}

      /// @name Additional helper functions
      /// @{
      bool empty() const {
         return m_size == 0;
      }

      size_type size() const {
         return m_size;
      }
      /// @}

   private:
      /// Size of the array that this object looks at
      size_type m_size;
      /// Pointer to the start of the array
      pointer m_ptr;

   }; // class vector

} // namespace detray::cuda

#endif // DETRAY_DATA_MODEL_VECTOR_HPP
