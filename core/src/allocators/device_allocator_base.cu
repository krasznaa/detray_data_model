/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "detraydm/allocators/device_allocator_base.hpp"
#include "detraydm/utils/cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime.h>

namespace detraydm::cuda {

   void* device_allocator_base::cuda_allocate( std::size_t nBytes ) {

      void* ptr = nullptr;
      DETRAYDM_CUDA_ERROR_CHECK( cudaMalloc( &ptr, nBytes ) );
      return ptr;
   }

   void device_allocator_base::cuda_deallocate( void* ptr ) {

      DETRAYDM_CUDA_ERROR_CHECK( cudaFree( ptr ) );
   }

} // namespace detraydm::cuda
