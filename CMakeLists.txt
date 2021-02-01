# Copyright (C) 2021 Attila Krasznahorkay.

# Set up the project.
cmake_minimum_required( VERSION 3.10 )
project( DetrayDataModel VERSION 0.0.1 LANGUAGES CXX CUDA )

# Set up the used C++ standard(s).
set( CMAKE_CXX_STANDARD 17 CACHE STRING "The (host) C++ standard to use" )
set( CMAKE_CUDA_STANDARD 14 CACHE STRING "The (CUDA) C++ standard to use" )

# Explicitly set the CUDA architecture to build for. Needed by CMake 3.18+.
set( CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING
   "CUDA architecture(s) to build device code for" )

# Standard CMake include(s).
include( CTest )
include( GNUInstallDirs )

# Build a (static) library with the data model code.
add_library( DetrayCudaDataModel
   core/cuda_error_check.cuh core/cuda_error_check.cu
   allocators/host_allocator_base.hpp allocators/host_allocator_base.cu
   allocators/host_allocator.hpp
   allocators/managed_allocator_base.hpp allocators/managed_allocator_base.cu
   allocators/managed_allocator.hpp
   detector/dummy_detector.hpp
   vector/vector.hpp )
target_include_directories( DetrayCudaDataModel PUBLIC
   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

# Build the test for using the allocator(s) in the host code.
add_executable( test_host_allocations tests/test_host_allocations.cpp )
target_link_libraries( test_host_allocations DetrayCudaDataModel )
add_test( NAME test_host_allocations
       COMMAND test_host_allocations )

# Build the test for using the custom vector in device code.
add_executable( test_vector_on_device tests/test_vector_on_device.cu )
target_link_libraries( test_vector_on_device DetrayCudaDataModel )
add_test( NAME test_vector_on_device
       COMMAND test_vector_on_device )

# Build a pure C++ test for the dummy detector type.
add_executable( test_detector tests/test_detector.cpp )
target_link_libraries( test_detector DetrayCudaDataModel )
add_test( NAME test_detector
       COMMAND test_detector )
