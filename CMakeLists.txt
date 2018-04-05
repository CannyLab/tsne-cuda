cmake_minimum_required (VERSION 2.8)
project(cuda-tsne)
set (cuda-tsne_VERSION_MAJOR 0)
set (cuda-tsne_VERSION_MINOR 1)
set (CMAKE_SKIP_RULE_DEPENDENCY TRUE)

# Fix for Roshan's Compiler
#-------------------------------------------------------------------------------
if (NOT CXX)
	set(CXX g++)
endif()
#-------------------------------------------------------------------------------


# CUDA Configuration
#-------------------------------------------------------------------------------

SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)
SET(CUDA_HOST_COMPILER ${CXX})
#-------------------------------------------------------------------------------
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; 
                    -O3 
                    -gencode=arch=compute_30,code=sm_30
                    -gencode=arch=compute_35,code=sm_35
                    -gencode=arch=compute_50,code=sm_50
                    -gencode=arch=compute_61,code=sm_61
                    -std=c++11
                    -g
                    -Xcompiler '-fPIC'
                    )
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


# Project Setup
#-------------------------------------------------------------------------------
include_directories(
    src/include
    ${CUDA_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/gtest/src/gtest/googletest/include
)

set(SOURCES
    src/naive_tsne.cu
    src/util/cuda_utils.cu
    src/util/distance_utils.cu
    src/util/math_utils.cu
    src/util/matrix_broadcast_utils.cu
    src/util/random_utils.cu
    src/util/reduce_utils.cu
    src/ext/pymodule_ext.cu
)
#-------------------------------------------------------------------------------

# Shared Library for Python Binding
#-------------------------------------------------------------------------------
cuda_add_library(pyctsne
                 SHARED
                 ${SOURCES})
target_link_libraries(pyctsne ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
set_target_properties(pyctsne PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_custom_command(TARGET pyctsne POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/src/python $<TARGET_FILE_DIR:pyctsne>/python)
add_custom_command(TARGET pyctsne POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pyctsne> $<TARGET_FILE_DIR:pyctsne>/python/pyctsne)

#-------------------------------------------------------------------------------


# Test Target
#-------------------------------------------------------------------------------
cuda_add_executable(
    tsne_test
    ${SOURCES}
    src/test/test.cu
)
add_dependencies(tsne_test gtest)
target_link_libraries(
    tsne_test
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
    NAME tsne_test
    COMMAND tsne_test
)
#-------------------------------------------------------------------------------
