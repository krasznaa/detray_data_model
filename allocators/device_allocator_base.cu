// Copyright (C) 2021 Attila Krasznahorkay.

// Local include(s).
#include "device_allocator_base.hpp"
#include "core/cuda_error_check.cuh"

// CUDA include(s).
#include <cuda_runtime.h>

namespace detray::cuda {

   void* device_allocator_base::cuda_allocate( std::size_t nBytes ) {

      void* ptr = nullptr;
      DETRAY_CUDA_ERROR_CHECK( cudaMalloc( &ptr, nBytes ) );
      return ptr;
   }

   void device_allocator_base::cuda_deallocate( void* ptr ) {

      DETRAY_CUDA_ERROR_CHECK( cudaFree( ptr ) );
   }

} // namespace detray::cuda
