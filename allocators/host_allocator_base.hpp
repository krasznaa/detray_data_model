// Copyright (C) 2021 Attila Krasznahorkay.
#ifndef DETRAY_DATA_MODEL_HOST_ALLOCATOR_BASE_HPP
#define DETRAY_DATA_MODEL_HOST_ALLOCATOR_BASE_HPP

// System include(s).
#include <cstddef>

namespace detray::cuda {

   /// Base class for @c detray::cuda::host_allocator<T>
   ///
   /// This is the class responsible for actually talking to CUDA.
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   class host_allocator_base {

   public:
      /// Allocate the specified number of bytes of memory
      static void* cuda_allocate( std::size_t nBytes );
      /// Deallocate a previously allocated block of memory
      static void cuda_deallocate( void* ptr );

   }; // class host_allocator_base

} // namespace detray::cuda

#endif // DETRAY_DATA_MODEL_HOST_ALLOCATOR_BASE_HPP
