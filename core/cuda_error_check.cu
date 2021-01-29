// Copyright (C) 2021 Attila Krasznahorkay.

// Local include(s).
#include "cuda_error_check.cuh"

// CUDA include(s).
#include <cuda_runtime.h>

// System include(s).
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace detray::cuda::details {

   void throw_error( cudaError_t errorCode, const char* expression,
                     const char* file, int line ) {

      // Create a nice error message.
      std::ostringstream errorMsg;
      errorMsg << file << ":" << line << " Failed to execute: " << expression
               << " (" << cudaGetErrorString( errorCode ) << ")";

      // Now throw a runtime error with this message.
      throw std::runtime_error( errorMsg.str() );
   }

} // namespace detray::cuda::details
