# Copyright (C) 2021 Attila Krasznahorkay. All rights reserved.

# Set up the project.
cmake_minimum_required( VERSION 3.10 )
project( DetrayDataModel VERSION 0.0.1 LANGUAGES CXX CUDA )

# Set up the used C++ standard(s).
set( CMAKE_CXX_STANDARD 17 CACHE STRING "The (host) C++ standard to use" )
set( CMAKE_CUDA_STANDARD 14 CACHE STRING "The (CUDA) C++ standard to use" )

# Standard CMake include(s).
include( GNUInstallDirs )

# Build a (static) library with the memory allocator code.
add_library( detrayCudaAllocators
   core/cuda_error_check.cuh core/cuda_error_check.cu
   allocators/managed_allocator_base.hpp allocators/managed_allocator_base.cu
   allocators/managed_allocator.hpp )
target_include_directories( detrayCudaAllocators PUBLIC
   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> )

# Build the test for using the allocator(s) in the host code.
add_executable( test_host_allocations tests/test_host_allocations.cpp )
target_link_libraries( test_host_allocations detrayCudaAllocators )
