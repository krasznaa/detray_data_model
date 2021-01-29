// Copyright (C) 2021 Attila Krasznahorkay.
#ifndef DETRAY_DATA_MODEL_HOST_ALLOCATOR_HPP
#define DETRAY_DATA_MODEL_HOST_ALLOCATOR_HPP

// Local include(s).
#include "host_allocator_base.hpp"

// System include(s).
#include <cstddef>
#include <type_traits>

namespace detray::cuda {

   /// CUDA host memory allocator to use with STL container types
   ///
   /// Making sure that the memory created for the STL container on the host
   /// would be page-locked. (Not allowed to be moved to swap, or any other part
   /// of the memory.)
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< typename TYPE >
   class host_allocator : public host_allocator_base {

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

   }; // class host_allocator

} // namespace detray::cuda

#endif // DETRAY_DATA_MODEL_HOST_ALLOCATOR_HPP
