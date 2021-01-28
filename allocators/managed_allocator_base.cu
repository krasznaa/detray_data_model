// Copyright (C) 2021 Attila Krasznahorkay. All rights reserved.

// Local include(s).
#include "managed_allocator_base.hpp"
#include "core/cuda_error_check.cuh"

// CUDA include(s).
#include <cuda_runtime.h>

namespace detray::cuda {

   void* managed_allocator_base::cuda_allocate( std::size_t nBytes ) {

      void* ptr = nullptr;
      DETRAY_CUDA_ERROR_CHECK( cudaMallocManaged( &ptr, nBytes ) );
      return ptr;
   }

   void managed_allocator_base::cuda_deallocate( void* ptr ) {

      DETRAY_CUDA_ERROR_CHECK( cudaFree( ptr ) );
   }

} // namespace detray::cuda
