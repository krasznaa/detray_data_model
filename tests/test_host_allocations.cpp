/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "detraydm/allocators/device_allocator.hpp"
#include "detraydm/allocators/host_allocator.hpp"
#include "detraydm/allocators/managed_allocator.hpp"
#include "detraydm/containers/device_vector.hpp"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <vector>

// Some helper variable(s).
static const std::size_t VEC_ELEMENTS = 100;

int main() {

   // Create a test object with the host allocator.
   std::vector< float, detraydm::cuda::host_allocator< float > > host_vector;

   // Make sure that it behaves correctly.
   for( std::size_t i = 0; i < VEC_ELEMENTS; ++i ) {
      host_vector.push_back( 1.0f * i );
   }
   assert( host_vector.size() == VEC_ELEMENTS );
   for( std::size_t i = 0; i < VEC_ELEMENTS; ++i ) {
      assert( std::abs( host_vector.at( i ) - 1.0f * i ) < 0.001 );
   }

   // Create a test object with the managed allocator.
   std::vector< float, detraydm::cuda::managed_allocator< float > >
      managed_vector;

   // Fill it with some data.
   for( std::size_t i = 0; i < VEC_ELEMENTS; ++i ) {
      managed_vector.push_back( 1.0f * i );
   }
   assert( managed_vector.size() == VEC_ELEMENTS );
   assert( std::abs( managed_vector[ 20 ] - 20.0f ) < 0.001 );

   // Create a "kernel vector".
   detraydm::cuda::device_vector< float >
      kernel_vector( managed_vector.size(), managed_vector.data() );
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

   // Create a "device vector". Note that one must not actually write directly
   // to such an object. It is really only used for memory management...
   std::vector< float, detraydm::cuda::device_allocator< float > >
      device_vector;
   // Change its size a couple of times, just to see that it would work.
   device_vector.resize( 20 );
   device_vector.reserve( 1000 );
   device_vector.resize( 500 );
   device_vector.resize( 5 );

   // Return gracefully.
   return 0;
}
