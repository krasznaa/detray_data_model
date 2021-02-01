// Copyright (C) 2021 Attila Krasznahorkay.

// Local include(s).
#include "detector/dummy_detector.hpp"
#include "vector/static_vector.hpp"

// System include(s).
#include <iostream>

/// Vector type to be used in the volumes.
template< typename T >
using static_volume_vector = detray::static_vector< T, 100 >;

int main() {

   // Create a detector object, and fill it with some nonsensical data.
   detray::dummy_detector<> hdet1;
   for( int i = 0; i < 10; ++i ) {
      hdet1.surfaces().emplace_back( i, i + 1 );
   }
   for( int i = 0; i < 5; ++i ) {
      hdet1.volumes().emplace_back();
      for( int j = i; j < 10; ++j ) {
         hdet1.volumes().back().m_surface_indices.push_back( j );
      }
   }

   /// Detector type with a fixed sized volume vector/array, on the host
   using host_fixed_detector = detray::dummy_detector< detray::host_vector,
                                                       static_volume_vector >;

   // Create a detector object with "fixed internal sizes".
   host_fixed_detector hdet2 = hdet1;

   // Access the internal data of the fixed-sized-detector.
   auto hdet_data = hdet2.data();

   //
   // Print some stuff, just to check whether things look as they should.
   //
   std::cout << "sizeof(host_detector::volume) = "
             << sizeof( detray::dummy_detector<>::volume ) << std::endl;
   std::cout << "sizeof(host_fixed_detector::volume) = "
             << sizeof( host_fixed_detector::volume ) << std::endl;
   std::cout << "m_surfaces = " << hdet_data.m_surfaces
             << ", m_nSurfaces = " << hdet_data.m_nSurfaces << std::endl;
   std::cout << "m_volumes = " << hdet_data.m_volumes
             << ", m_nVolumes = " << hdet_data.m_nVolumes << std::endl;

   for( std::size_t i = 0; i < hdet_data.m_nVolumes; ++i ) {
      std::cout << "Volume " << i << std::endl;
      std::cout << "Surface indices: ";
      const host_fixed_detector::volume& v = hdet_data.m_volumes[ i ];
      for( std::size_t index : v.m_surface_indices ) {
         std::cout << index << ", ";
      }
      std::cout << std::endl;
   }

   // Retrun gracefully.
   return 0;
}
