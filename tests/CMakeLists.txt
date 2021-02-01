# Detray Data Model project, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Test for the allocators.
add_executable( detraydmtest_host_allocations
   test_host_allocations.cpp )
target_link_libraries( detraydmtest_host_allocations detraydm::core )
add_test( NAME detraydmtest_host_allocations
       COMMAND detraydmtest_host_allocations )

# Build the test for using the custom vector in device code.
add_executable( detraydmtest_containers
   test_containers.cu )
target_link_libraries( detraydmtest_containers detraydm::core )
add_test( NAME detraydmtest_containers
       COMMAND detraydmtest_containers )

# Build a pure C++ test for the dummy detector type.
add_executable( detraydmtest_detector_on_host
   detector.hpp test_detector_on_host.cpp )
target_link_libraries( detraydmtest_detector_on_host detraydm::core )
add_test( NAME detraydmtest_detector_on_host
       COMMAND detraydmtest_detector_on_host )

# Build an "on device" test for the dummy detector type.
add_executable( detraydmtest_detector_on_device
   detector.hpp test_detector_on_device.cu )
target_link_libraries( detraydmtest_detector_on_device detraydm::core )
add_test( NAME detraydmtest_detector_on_device
       COMMAND detraydmtest_detector_on_device )
