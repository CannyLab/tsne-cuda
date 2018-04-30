cmake_minimum_required (VERSION 2.8)
project(cuda-tsne)
set (cuda-tsne_VERSION_MAJOR 0)
set (cuda-tsne_VERSION_MINOR 1)
set (CMAKE_SKIP_RULE_DEPENDENCY TRUE)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Fix for Roshan's Compiler
#-------------------------------------------------------------------------------
if (NOT CXX)
	set(CXX g++)
endif()
if (NOT NTHREADS)
    set(NTHREADS 6)
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
                    #-gencode=arch=compute_30,code=sm_30
                    #-gencode=arch=compute_35,code=sm_35
                    -gencode=arch=compute_50,code=sm_50
                    -gencode=arch=compute_61,code=sm_61
                    -std=c++11
                    -g
                    -Xcompiler '-fPIC'
                    -Xcompiler '-fopenmp'
                    -Xcompiler '-msse4'
                    -Xcompiler '-m64'
                    -Xcompiler '-mpopcnt'
                    
                    )
get_filename_component(CUDA_LIB_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)
find_library(CUDA_cudadevrt_LIBRARY cudadevrt PATHS ${CUDA_LIB_PATH})
#-------------------------------------------------------------------------------

# OpenMP configuration
#-------------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
#-------------------------------------------------------------------------------

# BLAS configuration
#-------------------------------------------------------------------------------
option(WITH_MKL "Build with MKL if ON (OpenBLAS if OFF)" ON)
if(WITH_MKL)
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIRS})
    set(BLAS_LIB ${MKL_LIBRARIES})
else()
    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(BLAS_LIB ${OpenBLAS_LIB})
endif()
#-------------------------------------------------------------------------------

# GTEST Configuration
#-------------------------------------------------------------------------------
include(ExternalProject)
ExternalProject_Add(gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG  release-1.8.0
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/gtest
  BUILD_COMMAND make gtest && make gtest_main
  INSTALL_COMMAND ""
)
#-------------------------------------------------------------------------------

# FAISS Configuration
#-------------------------------------------------------------------------------
ExternalProject_Add(faiss
  GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
  GIT_TAG  v1.2.1
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/faiss
  CMAKE_ARGS -DCMAKE_PREFIX_PATH=${source_dir};${binary_dir} -DWITH_MKL=${WITH_MKL} -DMKL_LIBRARIES=${DMKL_LIBRARIES} -DMKL_INCLUDE_DIRS=${MKL_INCLUDE_DIRS} -DOpenBLAS_LIB=${OpenBLAS_LIB} -DOpenBLAS_INCLUDE_DIR=${OpenBLAS_INCLUDE_DIR} -DBUILD_TEST=NO
  BUILD_COMMAND make faiss -j${NTHREADS} && make gpufaiss -j${NTHREADS}
  INSTALL_COMMAND ""
)
#-------------------------------------------------------------------------------

# Project Setup
#-------------------------------------------------------------------------------
include_directories(
    src/include
    ${CUDA_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/gtest/src/gtest/googletest/include
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/faiss/src/
)

link_directories(
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/faiss/src/faiss-build/lib/
    ${CUDA_LIB_PATH}
)

set(SOURCES
    src/naive_tsne.cu
    src/naive_tsne_cpu.cu
    src/util/data_utils.cu
    src/util/cuda_utils.cu
    src/util/thrust_utils.cu
    src/util/distance_utils.cu
    src/util/math_utils.cu
    src/util/matrix_broadcast_utils.cu
    src/util/random_utils.cu
    src/util/reduce_utils.cu
    src/ext/pymodule_ext.cu
    src/bh.cu
    src/bh_tsne.cu
    src/bh_tsne_ref.cu
)
#-------------------------------------------------------------------------------

# Shared Library for Python Binding
#-------------------------------------------------------------------------------

set(PYTHON_SOURCES 
    src/python/pyctsne/TSNE.py
)

cuda_add_library(pyctsne
                 SHARED
                 ${SOURCES})
add_dependencies(pyctsne gtest)
add_dependencies(pyctsne faiss)
target_link_libraries(pyctsne ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} zmq)
set_target_properties(pyctsne PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_custom_command(TARGET pyctsne POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/src/python $<TARGET_FILE_DIR:pyctsne>/python)
add_custom_command(TARGET pyctsne POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pyctsne> $<TARGET_FILE_DIR:pyctsne>/python/pyctsne)

ADD_CUSTOM_TARGET(copy_pyctsne DEPENDS ${PYTHON_SOURCES})
ADD_DEPENDENCIES(pyctsne copy_pyctsne)
add_dependencies(copy_pyctsne gtest)
add_dependencies(copy_pyctsne faiss)

#-------------------------------------------------------------------------------

# Test Target
#-------------------------------------------------------------------------------
cuda_add_executable(tsne_test
    ${SOURCES}
    src/test/test.cu
)
add_dependencies(tsne_test gtest)
add_dependencies(tsne_test faiss)
target_link_libraries(tsne_test
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/gtest/src/gtest-build/googlemock/gtest/libgtest.a
    ${CMAKE_CURRENT_BINARY_DIR}/third_party/gtest/src/gtest-build/googlemock/gtest/libgtest_main.a
    ${CUDA_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    #/usr/local/cuda-8.0/lib64/libcudaevrt.a
    ${CUDA_cudadevrt_LIBRARY}
    libfaiss.a
    libgpufaiss.a
    pthread
    zmq
    ${BLAS_LIB}
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

# Documentation
#-------------------------------------------------------------------------------
# first we can indicate the documentation build as an option and set it to ON by default
option(BUILD_DOC "Build documentation" OFF)

# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND AND BUILD_DOC)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
elseif(NOT DOXYGEN_FOUND AND BUILD_DOC)
    message("Doxygen need to be installed to generate the doxygen documentation")
else (DOXYGEN_FOUND AND BUILD_DOC)
    message("Not building documentation. To build documentation use -DBUILD_DOC=ON")
endif (DOXYGEN_FOUND AND BUILD_DOC)
