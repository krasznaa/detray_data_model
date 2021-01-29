// Copyright (C) 2021 Attila Krasznahorkay.
#ifndef DETRAY_DATA_MODEL_CUDA_ERROR_CHECK_CUH
#define DETRAY_DATA_MODEL_CUDA_ERROR_CHECK_CUH

/// Helper macro used in the CUDA plugin for checking @c cudaError_t type return
/// values.
#define DETRAY_CUDA_ERROR_CHECK( EXP )                                         \
   do {                                                                        \
      cudaError_t errorCode = EXP;                                             \
      if( errorCode != cudaSuccess ) {                                         \
        detray::cuda::details::throw_error( errorCode, #EXP, __FILE__,         \
                                            __LINE__ );                        \
      }                                                                        \
   } while( false )

namespace detray::cuda::details {

   /// Function used to print and throw a user-readable error if something breaks
   void throw_error( cudaError_t errorCode, const char* expression,
                     const char* file, int line );

} // namespace detray::cuda::details

#endif // DETRAY_DATA_MODEL_CUDA_ERROR_CHECK_CUH
