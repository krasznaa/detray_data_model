/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "detraydm/allocators/device_allocator_base.hpp"
#include "detraydm/memory/memory_manager.hpp"
#include "detraydm/memory/memory_manager_interface.hpp"

namespace detraydm::cuda {

   void* device_allocator_base::cuda_allocate( std::size_t nBytes ) {

      return memory_manager::instance().get().allocate( nBytes );
   }

   void device_allocator_base::cuda_deallocate( void* ptr ) {

      memory_manager::instance().get().deallocate( ptr );
   }

} // namespace detraydm::cuda
