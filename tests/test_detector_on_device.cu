/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "detector.hpp"
#include "detraydm/allocators/device_allocator.hpp"
#include "detraydm/allocators/managed_allocator.hpp"
#include "detraydm/containers/const_device_vector.hpp"
#include "detraydm/containers/device_vector.hpp"
#include "detraydm/containers/static_vector.hpp"
#include "detraydm/utils/cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime.h>

// System include(s).
#include <iostream>
#include <vector>

/// Helper type for managing the geometry memory on the device
template< template< typename > class volume_vector_t >
struct device_detector_data :
   public detraydm::test::detector_data< volume_vector_t > {

   /// Base type for the struct
   typedef detraydm::test::detector_data< volume_vector_t > base_type;

   /// Type of the vector managing the device memory for the surfaces
   using surface_vector_type =
      std::vector< typename base_type::surface_type,
                   detraydm::cuda::device_allocator< typename base_type::surface_type > >;
   /// Type of the vector managing the device memory for the volumes
   using volume_vector_type =
      std::vector< typename base_type::volume_type,
                   detraydm::cuda::device_allocator< typename base_type::volume_type > >;

   /// Vector managing the device memory for the surfaces
   surface_vector_type m_surfacesVec;
   /// Vector managing the device memory for the volumes
   volume_vector_type m_volumesVec;

}; // struct device_detector_data

/// Helper function moving the data of the detector to the device
template< template< typename > class volume_vector_type >
void copy_host_to_device( const detraydm::test::detector_data< volume_vector_type >& hostData,
                          device_detector_data< volume_vector_type >& deviceData ) {

   // Copy the values of the primitive variables.
   deviceData.m_nSurfaces = hostData.m_nSurfaces;
   deviceData.m_nVolumes  = hostData.m_nVolumes;

   /// Helper size of a surface object in memory.
   static constexpr std::size_t surface_size =
      sizeof( typename detraydm::test::detector_data< volume_vector_type >::surface_type );
   /// Helper size of a volume object in memory.
   static constexpr std::size_t volume_size =
      sizeof( typename detraydm::test::detector_data< volume_vector_type >::volume_type );

   // Copy the arrays to the device.
   deviceData.m_surfacesVec.resize( deviceData.m_nSurfaces );
   deviceData.m_surfaces = deviceData.m_surfacesVec.data();
   DETRAYDM_CUDA_ERROR_CHECK( cudaMemcpy( deviceData.m_surfacesVec.data(),
                                          hostData.m_surfaces,
                                          hostData.m_nSurfaces * surface_size,
                                          cudaMemcpyHostToDevice ) );
   deviceData.m_volumesVec.resize( deviceData.m_nVolumes );
   deviceData.m_volumes = deviceData.m_volumesVec.data();
   DETRAYDM_CUDA_ERROR_CHECK( cudaMemcpy( deviceData.m_volumesVec.data(),
                                          hostData.m_volumes,
                                          hostData.m_nVolumes * volume_size,
                                          cudaMemcpyHostToDevice ) );
   return;
}

/// Kernel doing something nonsensical with the detector geometry
template< template< typename > class volume_vector_type >
__global__
void testDetectorKernel( std::size_t size, int* array,
                         detraydm::test::detector_data< volume_vector_type > detData ) {

   // Skip invalid elements.
   const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if( i >= size ) {
      return;
   }

   // Create a smart detector object on top of the data in global device memory.
   using device_fixed_detector =
      detraydm::test::detector< detraydm::cuda::const_device_vector,
                                volume_vector_type >;
   device_fixed_detector det( detData );

   // Construct a helper object on top of the array.
   detraydm::cuda::device_vector< int > vec( size, array );

   // Do something nonsensical.
   const std::size_t volume_index = i % det.volumes().size();
   const std::size_t surface_index =
      i % det.volumes().at( volume_index ).m_surface_indices.size();
   vec.at( i ) +=
      det.volumes().at( volume_index ).m_surface_indices.at( surface_index );
   return;
}

/// Helper operator for printing the contents of vectors
template< typename T, typename A >
std::ostream& operator<<( std::ostream& out, const std::vector< T, A >& vec ) {
   out << "[";
   for( std::size_t i = 0; i < vec.size(); ++i ) {
      out << vec[ i ];
      if( i + 1 < vec.size() ) {
         out << ", ";
      }
   }
   out << "]";
   return out;
}

/// Vector type to be used in the volumes.
template< typename T >
using static_volume_vector = detraydm::cuda::static_vector< T, 20 >;

int main() {

   /// Detector type with a fixed sized volume vector/array, on the host
   using host_fixed_detector = detraydm::test::detector< detraydm::host_vector,
                                                         static_volume_vector >;

   // Create a detector object, and fill it with some nonsensical data.
   host_fixed_detector hdet;
   for( int i = 0; i < 10; ++i ) {
      hdet.surfaces().push_back( { i, i + 1 } );
   }
   for( int i = 0; i < 5; ++i ) {
      hdet.volumes().emplace_back();
      for( int j = i; j < 10; ++j ) {
         hdet.volumes().back().m_surface_indices.push_back( j );
      }
   }

   // Copy the detector's payload to the device.
   device_detector_data< static_volume_vector > deviceData;
   copy_host_to_device( hdet.data(), deviceData );

   // Allocate a simple array that the kernel could write to.
   std::vector< int, detraydm::cuda::managed_allocator< int > >
      managed_vector( 100 );

   // Launch a dummy kernel that makes some use of the detector geometry.
   testDetectorKernel<<< managed_vector.size(), 1 >>>( managed_vector.size(),
                                                       managed_vector.data(),
                                                       deviceData );
   DETRAYDM_CUDA_ERROR_CHECK( cudaGetLastError() );
   DETRAYDM_CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

   // Print the results.
   std::cout << managed_vector << std::endl;

   // Return gracefully.
   return 0;
}
