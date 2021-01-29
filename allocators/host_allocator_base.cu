// Copyright (C) 2021 Attila Krasznahorkay.

// Local include(s).
#include "host_allocator_base.hpp"
#include "core/cuda_error_check.cuh"

// CUDA include(s).
#include <cuda_runtime.h>

namespace detray::cuda {

   void* host_allocator_base::cuda_allocate( std::size_t nBytes ) {

      void* ptr = nullptr;
      DETRAY_CUDA_ERROR_CHECK( cudaMallocHost( &ptr, nBytes ) );
      return ptr;
   }

   void host_allocator_base::cuda_deallocate( void* ptr ) {

      DETRAY_CUDA_ERROR_CHECK( cudaFreeHost( ptr ) );
   }

} // namespace detray::cuda
