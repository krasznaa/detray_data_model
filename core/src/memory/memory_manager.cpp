/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "detraydm/memory/memory_manager.hpp"
#include "detraydm/memory/memory_manager_interface.hpp"
#include "detraydm/memory/cuda/arena_memory_manager.hpp"

namespace detraydm {

   /// The destructor needs to be explicitly implemented, so that
   /// @c detraydm::memory_manager_interface could be forward-declared in
   /// @c memory_manager.hpp.
   memory_manager::~memory_manager() {

   }

   memory_manager& memory_manager::instance() {

      static memory_manager instance;
      return instance;
   }

   void memory_manager::set( std::unique_ptr< memory_manager_interface > mgr ) {

      m_mgr = std::move( mgr );
      return;
   }

   memory_manager_interface& memory_manager::get() const {

      return *m_mgr;
   }

   memory_manager::memory_manager()
   : m_mgr( new cuda::arena_memory_manager() ) {

   }

} // namespace detraydm
