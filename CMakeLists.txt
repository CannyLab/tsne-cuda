cmake_minimum_required (VERSION 2.8)
project(cuda-tsne)
set (cuda-tsne_VERSION_MAJOR 0)
set (cuda-tsne_VERSION_MINOR 1)


# CUDA Configuration
#-------------------------------------------------------------------------------
find_package(CUDA REQUIRED)
#-------------------------------------------------------------------------------
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; 
                    -O3 
                    -gencode=arch=compute_30,code=sm_30
                    -gencode=arch=compute_35,code=sm_35
                    -gencode=arch=compute_50,code=sm_50
                    -gencode=arch=compute_61,code=sm_61
                    -ccbin g++
                    -std=c++11 
                    -g)
#-------------------------------------------------------------------------------

# GTEST Configuration
#-------------------------------------------------------------------------------
include(ExternalProject)
ExternalProject_Add(gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG  release-1.8.0
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/gtest
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(gtest source_dir binary_dir)
#-------------------------------------------------------------------------------


# Utilities Target TODO: Breakout file
#-------------------------------------------------------------------------------
include_directories(
    src/include
    ${CUDA_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/gtest/src/gtest/googletest/include
)

cuda_add_executable(
    tsne_utils
    src/tsne_utils.cu
    src/Utilities.cu
)
target_link_libraries(tsne_utils ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
#-------------------------------------------------------------------------------

# Test Target
#-------------------------------------------------------------------------------
cuda_add_executable(
    runUnitTests
    src/test/test.cu
)
add_dependencies(runUnitTests gtest)
target_link_libraries(
    runUnitTests
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/gtest/src/gtest-build/googlemock/gtest/libgtest.a
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/gtest/src/gtest-build/googlemock/gtest/libgtest_main.a
    ${CUDA_LIBRARIES} 
    ${CUDA_CUBLAS_LIBRARIES}
    pthread
)
#-------------------------------------------------------------------------------

# Testing
#-------------------------------------------------------------------------------
enable_testing()
add_test(
    NAME runUnitTests
    COMMAND runUnitTests
)
#-------------------------------------------------------------------------------