/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "detraydm/memory/memory_manager_interface.hpp"

// System include(s).
#include <list>
#include <vector>

namespace detraydm::cuda {

   /// A very simple CUDA memory manager using CUDA memory functions directly
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   class direct_memory_manager : public memory_manager_interface {

   public:
      /// Constructor, allocating the default amount of memory
      direct_memory_manager();
      /// Destructor, freeing up all allocations
      ~direct_memory_manager();

      /// Default device, the one CUDA is set up to use at a given time
      static constexpr int DEFAULT_DEVICE = -1;

      /// @name Functions inherited from @c detraydm::memory_manager_interface
      /// @{

      /// Set the amount of memory to use on a particular device
      void set_maximum_capacity( std::size_t sizeInBytes, int device ) override;

      /// Get the amount of memory still available on a specific device
      std::size_t available_memory( int device ) const override;

      /// Get a pointer to an available memory block on a specific device
      void* allocate( std::size_t sizeInBytes, int device ) override;

      /// Deallocate a specific memory block
      void deallocate( void* ptr ) override;

      /// Reset all allocations on a given device
      void reset( int device ) override;

      /// @}

   private:
      /// Struct describing the state of the memory allocation on a particular
      /// device
      struct device_memory {
         /// List of memory allocations on the device
         std::list< void* > m_ptrs;
      };

      /// Create a valid device ID from what the user provided
      static void get_device( int& device );
      /// Access the object describing the memory allocation on a specific
      /// device
      device_memory& get_device_memory( int& device );

      /// Object holding information about memory allocations on all devices
      std::vector< device_memory > m_memory;

   }; // class direct_memory_manager

} // namespace detraydm::cuda
