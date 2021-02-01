/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Detray Data Model include(s).
#include "detraydm/utils/cuda_types.hpp"

/// Helper macro used for checking @c cudaError_t type return values.
#ifdef __CUDACC__
#   define DETRAYDM_CUDA_ERROR_CHECK( EXP )                                    \
   do {                                                                        \
      cudaError_t errorCode = EXP;                                             \
      if( errorCode != cudaSuccess ) {                                         \
        detraydm::cuda::details::throw_error( errorCode, #EXP, __FILE__,       \
                                              __LINE__ );                      \
      }                                                                        \
   } while( false )
#else
#   define DETRAYDM_CUDA_ERROR_CHECK( EXP ) do {} while( false )
#endif // __CUDACC__

/// Helper macro used for running a CUDA function when not caring about its results
#ifdef __CUDACC__
#   define DETRAYDM_CUDA_ERROR_IGNORE( EXP )                                   \
   do {                                                                        \
      EXP;                                                                     \
   } while( false )
#else
#   define DETRAYDM_CUDA_ERROR_IGNORE( EXP ) do {} while( false )
#endif // __CUDACC__

namespace detraydm::cuda::details {

   /// Function used to print and throw a user-readable error if something breaks
   void throw_error( cudaError_t errorCode, const char* expression,
                     const char* file, int line );

} // namespace detraydm::cuda::details
