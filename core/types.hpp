// Copyright (C) 2021 Attila Krasznahorkay. All rights reserved.
#ifndef DETRAY_DATA_MODEL_TYPES_HPP
#define DETRAY_DATA_MODEL_TYPES_HPP

/// Macro for declaring a device function
#ifdef __CUDACC__
#   define DETRAY_DEVICE __device__
#else
#   define DETRAY_DEVICE
#endif // __CUDACC__

/// Macro for declaring a host function
#ifdef __CUDACC__
#   define DETRAY_HOST __host__
#else
#   define DETRAY_HOST
#endif // __CUDACC__

/// Macro for declaring a host+device function
#ifdef __CUDACC__
#   define DETRAY_HOST_AND_DEVICE __host__ __device__
#else
#   define DETRAY_HOST_AND_DEVICE
#endif // __CUDACC__

#endif // DETRAY_DATA_MODEL_TYPES_HPP
