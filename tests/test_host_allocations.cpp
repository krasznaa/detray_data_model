// Copyright (C) 2021 Attila Krasznahorkay. All rights reserved.

// Local include(s).
#include "allocators/managed_allocator.hpp"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <vector>

int main() {

   // Create a test object.
   std::vector< float, detray::cuda::managed_allocator< float > >
      managed_vector;

   for( int i = 0; i < 100; ++i ) {
      managed_vector.push_back( 1.0f * i );
   }

   assert( managed_vector.size() == 100 );
   assert( std::abs( managed_vector[ 20 ] - 20.0f ) < 0.001 );

   // Return gracefully.
   return 0;
}
