// Copyright (C) 2021 Attila Krasznahorkay.

// Local include(s).
#include "allocators/host_allocator.hpp"
#include "allocators/managed_allocator.hpp"
#include "vector/device_vector.hpp"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <vector>

// Some helper variable(s).
static const std::size_t VEC_ELEMENTS = 100;

int main() {

   // Create a test object with the host allocator.
   std::vector< float, detray::cuda::host_allocator< float > > host_vector;

   // Make sure that it behaves correctly.
   for( std::size_t i = 0; i < VEC_ELEMENTS; ++i ) {
      host_vector.push_back( 1.0f * i );
   }
   assert( host_vector.size() == VEC_ELEMENTS );
   for( std::size_t i = 0; i < VEC_ELEMENTS; ++i ) {
      assert( std::abs( host_vector.at( i ) - 1.0f * i ) < 0.001 );
   }

   // Create a test object with the managed allocator.
   std::vector< float, detray::cuda::managed_allocator< float > >
      managed_vector;

   // Fill it with some data.
   for( std::size_t i = 0; i < VEC_ELEMENTS; ++i ) {
      managed_vector.push_back( 1.0f * i );
   }
   assert( managed_vector.size() == VEC_ELEMENTS );
   assert( std::abs( managed_vector[ 20 ] - 20.0f ) < 0.001 );

   // Create a "kernel vector".
   detray::cuda::device_vector< float > kernel_vector( managed_vector.size(),
                                                       managed_vector.data() );
   // Test its most basic features.
   assert( kernel_vector.size() == VEC_ELEMENTS );
   for( std::size_t i = 0; i < VEC_ELEMENTS; ++i ) {
      assert( std::abs( kernel_vector.at( i ) - 1.0f * i ) < 0.001 );
      assert( std::abs( kernel_vector[ i ] - 1.0f * i ) < 0.001 );
   }
   // Test that its iterators work.
   {
      auto itr = kernel_vector.begin();
      auto end = kernel_vector.end();
      for( std::size_t i = 0; itr != end; ++itr, ++i ) {
         assert( std::abs( *itr - 1.0f * i ) < 0.001 );
      }
   }
   {
      auto ritr = kernel_vector.rbegin();
      auto rend = kernel_vector.rend();
      for( std::size_t i = VEC_ELEMENTS - 1; ritr != rend; ++ritr, --i ) {
         assert( std::abs( *ritr - 1.0f * i ) < 0.001 );
      }
   }
   // Test that range type loops work with it.
   std::size_t i = 0;
   for( float value : kernel_vector ) {
      assert( std::abs( value - 1.0f * i ) < 0.001 );
      ++i;
   }

   // Return gracefully.
   return 0;
}
