/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "detraydm/allocators/device_allocator_base.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace detraydm::cuda {

   /// CUDA managed memory allocator to use with STL container types
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< typename TYPE >
   class device_allocator : public device_allocator_base {

   public:
      /// @name Type definitions that need to be provided by the allocator
      /// @{
      typedef std::size_t    size_type;
      typedef std::ptrdiff_t difference_type;
      typedef TYPE*          pointer;
      typedef const TYPE*    const_pointer;
      typedef TYPE&          reference;
      typedef const TYPE&    const_reference;
      typedef TYPE           value_type;
      /// @}

      /// @name "Behaviour declarations" for the allocator
      /// @{
      typedef std::true_type propagate_on_container_move_assignment;
      typedef std::true_type is_always_equal;
      /// @}

      /// Allocate a requested amount of memory
      pointer allocate( size_type n, const void* = nullptr ) {
         return static_cast< pointer >(
            cuda_allocate( n * sizeof( value_type ) ) );
      }

      /// Deallocate a previously allocated block of memory
      void deallocate( pointer ptr, size_type ) {
         cuda_deallocate( ptr );
      }

      /// Prevent writing anything to the allocated area after the allocation
      ///
      /// This prevents for instance @c std::vector<T>::resize from trying to
      /// initialise the allocated memory to some default values.
      ///
      template< typename U, typename... Args >
      void construct( U*, Args&&... ) {}

   }; // class device_allocator

} // namespace detraydm::cuda
