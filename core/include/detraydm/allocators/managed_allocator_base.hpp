/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// System include(s).
#include <cstddef>

namespace detraydm::cuda {

   /// Base class for @c detray::cuda::managed_allocator<T>
   ///
   /// This is the class responsible for actually talking to CUDA.
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   class managed_allocator_base {

   public:
      /// Allocate the specified number of bytes of memory
      static void* cuda_allocate( std::size_t nBytes );
      /// Deallocate a previously allocated block of memory
      static void cuda_deallocate( void* ptr );

   }; // class managed_allocator_base

} // namespace detraydm::cuda
