// Copyright (C) 2021 Attila Krasznahorkay.

// Local include(s).
#include "core/cuda_error_check.cuh"
#include "allocators/managed_allocator.hpp"
#include "vector/vector.hpp"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <vector>

// Some helper variable(s).
static const std::size_t VEC_ELEMENTS = 100;

/// Kernel adding a specified value to each element of a simple vector.
__global__
void addToElements( std::size_t size, float* ptr, float value ) {

   // Skip invalid elements.
   const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if( i >= size ) {
      return;
   }

   // Construct a helper object on top of the array.
   detray::cuda::vector< float > vec( size, ptr );

   // Modify one array element with the help of the custom vector object.
   vec.at( i ) += value;
   return;
}

int main() {

   // Create a test object.
   std::vector< float, detray::cuda::managed_allocator< float > >
      managed_vector;
   for( std::size_t i = 0; i < VEC_ELEMENTS; ++i ) {
      managed_vector.push_back( 1.0f * i );
   }

   // Constant to increase the vector elements by.
   static const float CONSTANT = 20.0f;

   // Launch a kernel on top of this vector.
   addToElements<<< managed_vector.size(), 1 >>>( managed_vector.size(),
                                                  managed_vector.data(),
                                                  CONSTANT );
   DETRAY_CUDA_ERROR_CHECK( cudaGetLastError() );
   DETRAY_CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

   // Make sure that the memory modification worked.
   for( std::size_t i = 0; i < VEC_ELEMENTS; ++i ) {
      assert( std::abs( managed_vector.at( i ) - 1.0f * i -
                        CONSTANT ) < 0.001 );
   }

   // Return gracefully.
   return 0;
}
