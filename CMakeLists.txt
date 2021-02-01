# Detray Data Model project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the project.
cmake_minimum_required( VERSION 3.10 )
project( detray_data_model VERSION 0.0.1 LANGUAGES CXX CUDA )

# Standard CMake include(s).
include( GNUInstallDirs )

# Set up the used C++ standard(s).
set( CMAKE_CXX_STANDARD 17 CACHE STRING "The (host) C++ standard to use" )
set( CMAKE_CUDA_STANDARD 14 CACHE STRING "The (CUDA) C++ standard to use" )

# Explicitly set the CUDA architecture to build for. Needed by CMake 3.18+.
set( CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING
   "CUDA architecture(s) to build device code for" )

# Make sure that in debug mode the CUDA device code would also get debug
# symbols attached.
set( CMAKE_CUDA_FLAGS_DEBUG "-G" )

# Explicitly set the output directory for the binaries.
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}" CACHE PATH
   "Directory for the built binaries" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH
   "Directory for the built libraries" )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH
   "Directory for the built static libraries" )

# Add the sub-directories.
add_subdirectory( core )

# Export the configuration of the project.
include( CMakePackageConfigHelpers )
set( CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/detraydm" )
install( EXPORT detraydm
   NAMESPACE "detraydm::"
   FILE "detraydm-config-targets.cmake"
   DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )
configure_package_config_file(
   "${CMAKE_CURRENT_SOURCE_DIR}/cmake/detraydm-config.cmake.in"
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/detraydm-config.cmake"
   INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
   PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
             CMAKE_INSTALL_CMAKEDIR
   NO_CHECK_REQUIRED_COMPONENTS_MACRO )
write_basic_package_version_file(
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/detraydm-config-version.cmake"
   COMPATIBILITY "AnyNewerVersion" )
install( FILES
"${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/detraydm-config.cmake"
   "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/detraydm-config-version.cmake"
   DESTINATION "${CMAKE_INSTALL_CMAKEDIR}" )

# Set up the test(s).
include( CTest )
if( BUILD_TESTING )
   add_subdirectory( tests )
endif()
