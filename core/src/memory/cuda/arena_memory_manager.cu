/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "detraydm/memory/cuda/arena_memory_manager.hpp"
#include "detraydm/utils/cuda_error_handling.hpp"

// CUDA include(s).
#include <cuda_runtime.h>

namespace detraydm::cuda {

   arena_memory_manager::arena_memory_manager() {

      // Start out with a small-ish allocation.
      set_maximum_capacity( 200 * 1024l * 1024l, DEFAULT_DEVICE );
   }

   arena_memory_manager::~arena_memory_manager() {

      // Free all the allocated memory.
      for( device_memory& mem : m_memory ) {
         if( mem.m_ptr == nullptr ) {
            continue;
         }
         DETRAYDM_CUDA_ERROR_CHECK( cudaFree( mem.m_ptr ) );
      }
   }

   void arena_memory_manager::set_maximum_capacity( std::size_t sizeInBytes,
                                                    int device ) {

      // Get the object responsible for this device.
      device_memory& mem = get_device_memory( device );

      // De-allocate any previously allocated memory.
      if( mem.m_ptr ) {
         DETRAYDM_CUDA_ERROR_CHECK( cudaFree( mem.m_ptr ) );
      }

      // Allocate the newly requested amount.
      DETRAYDM_CUDA_ERROR_CHECK( cudaSetDevice( device ) );
      DETRAYDM_CUDA_ERROR_CHECK( cudaMalloc( &( mem.m_ptr ), sizeInBytes ) );

      // Set up the internal state of the object correctly.
      mem.m_size = sizeInBytes;
      mem.m_nextAllocation = mem.m_ptr;
      return;
   }

   std::size_t arena_memory_manager::available_memory( int device ) const {

      // Get a valid device.
      get_device( device );

      // Make sure that memory was allocated on the requested device.
      if( m_memory.size() <= static_cast< std::size_t >( device ) ) {
         throw std::bad_alloc();
      }
      const device_memory& mem = m_memory[ device ];

      // Return the requested information.
      return ( mem.m_size - ( mem.m_nextAllocation - mem.m_ptr ) );
    }

   void* arena_memory_manager::allocate( std::size_t sizeInBytes, int device ) {

      // Get the object responsible for this device.
      device_memory& mem = get_device_memory( device );

      // We already know what we want to return...
      void* result = mem.m_nextAllocation;

      // Make sure that all addresses given out are 8-byte aligned.
      static constexpr std::size_t ALIGN_SIZE = 8;
      const std::size_t misalignment = sizeInBytes % ALIGN_SIZE;
      const std::size_t padding =
         ( ( misalignment != 0 ) ? ( ALIGN_SIZE - misalignment ) : 0 );

      // Increment the internal pointer.
      mem.m_nextAllocation += sizeInBytes + padding;
      // And make sure that we didn't run out of memory.
      if( mem.m_nextAllocation - mem.m_ptr >= mem.m_size ) {
         throw std::bad_alloc();
      }

      // Apparently everything is okay.
      return result;
   }

   void arena_memory_manager::deallocate( void* ) {

      // This memory manager does not do piecewise de-allocations.
      return;
   }

   void arena_memory_manager::reset( int device ) {

      // Get the object responsible for this device.
      device_memory& mem = get_device_memory( device );

      // Note down how much memory was used in total until the reset.
      mem.m_maxUsage = std::max( mem.m_maxUsage,
                                 mem.m_nextAllocation - mem.m_ptr );

      // Return the internal pointer to its startout location.
      mem.m_nextAllocation = mem.m_ptr;
      return;
   }

   void arena_memory_manager::get_device( int& device ) {

      // If the user didn't ask for a specific device, use the one currently
      // used by CUDA.
      if( device == DEFAULT_DEVICE ) {
         DETRAYDM_CUDA_ERROR_CHECK( cudaGetDevice( &device ) );
      }
      return;
   }

   arena_memory_manager::device_memory&
   arena_memory_manager::get_device_memory( int& device ) {

      // Get a valid device.
      get_device( device );

      // Make sure that the internal storage variable is large enough.
      if( static_cast< std::size_t >( device ) >= m_memory.size() ) {
         m_memory.resize( device + 1 );
      }

      // Return the requested object.
      return m_memory[ device ];
   }

} // namespace detraydm::cuda
