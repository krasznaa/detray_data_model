# Detray Data Model project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the build of the (static) data model library.
add_library( detraydm_core
   # STL allocators.
   "include/detraydm/allocators/device_allocator_base.hpp"
   "include/detraydm/allocators/device_allocator.hpp"
   "src/allocators/device_allocator_base.cpp"
   "include/detraydm/allocators/host_allocator_base.hpp"
   "include/detraydm/allocators/host_allocator.hpp"
   "src/allocators/host_allocator_base.cu"
   "include/detraydm/allocators/managed_allocator_base.hpp"
   "include/detraydm/allocators/managed_allocator.hpp"
   "src/allocators/managed_allocator_base.cu"
   # STL mimicking containers.
   "include/detraydm/containers/const_device_vector.hpp"
   "include/detraydm/containers/device_vector.hpp"
   "include/detraydm/containers/host_vector.hpp"
   "include/detraydm/containers/static_vector.hpp"
   # Memory management.
   "include/detraydm/memory/memory_manager_interface.hpp"
   "include/detraydm/memory/memory_manager.hpp"
   "src/memory/memory_manager.cpp"
   "include/detraydm/memory/cuda/arena_memory_manager.hpp"
   "src/memory/cuda/arena_memory_manager.cu"
   "include/detraydm/memory/cuda/direct_memory_manager.hpp"
   "src/memory/cuda/direct_memory_manager.cu"
   # Utilities.
   "include/detraydm/utils/cuda_error_handling.hpp"
   "src/utils/cuda_error_handling.cu"
   "include/detraydm/utils/cuda_types.hpp" )
target_include_directories( detraydm_core PUBLIC
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

# Make sure that the library is available as "detraydm::core" in every
# situation.
set_target_properties( detraydm_core PROPERTIES
   EXPORT_NAME "core" )
add_library( detraydm::core ALIAS detraydm_core )

# Set up the installation of the library.
install( TARGETS detraydm_core
   EXPORT detraydm
   ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" )
install( DIRECTORY "include/"
   DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )
