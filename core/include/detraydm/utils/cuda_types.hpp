/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// CUDA include(s).
#ifdef __CUDACC__
#   include <cuda_runtime.h>
#else
typedef void* cudaStream_t; ///< Dummy type for @c cudaStream_t in C++ code
typedef int cudaError_t; ///< Dummy type for @c cudaError_t in C++ code
#endif // __CUDACC__


/// Macro for declaring a device function
#ifdef __CUDACC__
#   define DETRAYDM_CUDA_DEVICE __device__
#else
#   define DETRAYDM_CUDA_DEVICE
#endif // __CUDACC__

/// Macro for declaring a host function
#ifdef __CUDACC__
#   define DETRAYDM_CUDA_HOST __host__
#else
#   define DETRAYDM_CUDA_HOST
#endif // __CUDACC__

/// Macro for declaring a host+device function
#ifdef __CUDACC__
#   define DETRAYDM_CUDA_HOST_AND_DEVICE __host__ __device__
#else
#   define DETRAYDM_CUDA_HOST_AND_DEVICE
#endif // __CUDACC__
