/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "detraydm/utils/cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime.h>

// System include(s).
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace detraydm::cuda::details {

   void throw_error( cudaError_t errorCode, const char* expression,
                     const char* file, int line ) {

      // Create a nice error message.
      std::ostringstream errorMsg;
      errorMsg << file << ":" << line << " Failed to execute: " << expression
               << " (" << cudaGetErrorString( errorCode ) << ")";

      // Now throw a runtime error with this message.
      throw std::runtime_error( errorMsg.str() );
   }

} // namespace detraydm::cuda::details
