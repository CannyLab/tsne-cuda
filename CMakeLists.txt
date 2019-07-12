cmake_minimum_required (VERSION 3.0)
project(tsnecuda)
set (CMAKE_PROJECT_VERSION 2)
set (CMAKE_PROJECT_VERSION_MAJOR 1)
set (CMAKE_PROJECT_VERSION_MINOR 0)
set (CMAKE_PROJECT_VERSION_PATH 0)
set (CMAKE_SKIP_RULE_DEPENDENCY TRUE)
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Options
#-------------------------------------------------------------------------------
option(WITH_MKL "Build with MKL if ON (OpenBLAS if OFF)" OFF)
option(BUILD_TEST "Build tests if ON" OFF)
option(BUILD_PYTHON "Build python if ON" ON)
option(WITH_ZMQ "Build with interactive viz if ON" OFF)
option(WITH_FAISS_GPU_STANDALONE "Build standalone gpufaiss library ON" OFF)

if(NOT WITH_ZMQ)
    ADD_DEFINITIONS(-DNO_ZMQ)
    message("-- Not building with ZMQ. Interactive visualization disabled. To build with ZMQ use -DWITH_ZMQ=ON")
endif()

# Do we want to build FAISS as a standalone library as well.
if(NOT WITH_FAISS_GPU_STANDALONE)
    message("-- Not building standalone gpufaiss lib. To build gpufaiss standalone use -DWITH_FAISS_GPU_STANDALONE=ON")
endif()
#-------------------------------------------------------------------------------

# Build Number Generation
#-------------------------------------------------------------------------------
SET(BUILD_NUMBER 0)
SET(VERSION_STRING "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}")
find_package(Git)
if(GIT_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --match INITIAL_BUILD
            OUTPUT_VARIABLE DESCRIBE_BUILD
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+" BUILD_NUMBER ${DESCRIBE_BUILD})
endif()
ADD_DEFINITIONS(-DBUILD_NUMBER=${BUILD_NUMBER})
ADD_DEFINITIONS(-DVERSION_STRING=${VERSION_STRING})
#-------------------------------------------------------------------------------


# CXX Configuration
#-------------------------------------------------------------------------------
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#-------------------------------------------------------------------------------

# CUDA Configuration
#-------------------------------------------------------------------------------
SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)
SET(CUDA_HOST_COMPILER ${CXX})
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                    -O3
                    -Xptxas -dlcm=cg
                    -gencode=arch=compute_30,code=sm_30
                    -gencode=arch=compute_35,code=sm_35
                    -gencode=arch=compute_50,code=sm_50
                    -gencode=arch=compute_52,code=sm_52
                    -gencode=arch=compute_60,code=sm_60
                    -gencode=arch=compute_61,code=sm_61
                    -gencode=arch=compute_70,code=sm_70
                    -std=c++11
                    -Xcompiler '-O3'
                    -Xcompiler '-fPIC'
                    -Xcompiler '-fopenmp'
                    -Xcompiler '-msse4'
                    -Xcompiler '-m64'
                    -Xcompiler '-mpopcnt'
                    -Xcompiler '-g'

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
add_subdirectory(third_party/glog)
#-------------------------------------------------------------------------------

# GTEST Configuration
#-------------------------------------------------------------------------------
add_subdirectory(third_party/gtest)
#-------------------------------------------------------------------------------

# FAISS Configuration
#-------------------------------------------------------------------------------
add_subdirectory(third_party/faiss)
if(WITH_FAISS_GPU_STANDALONE)
    add_subdirectory(third_party/faiss/gpu)
endif()

# specify header and cpp files
file(GLOB_RECURSE faiss_gpu_headers third_party/faiss/gpu/*.h)
file(GLOB_RECURSE faiss_gpu_cpp third_party/faiss/gpu/*.cpp)
file(GLOB_RECURSE faiss_gpu_cuh third_party/faiss/gpu/*.cuh)
file(GLOB_RECURSE faiss_gpu_cu third_party/faiss/gpu/*.cu)
#-------------------------------------------------------------------------------

# Project Setup
#-------------------------------------------------------------------------------
include_directories(
    src/
    src/include
    ${CUDA_INCLUDE_DIRECTORIES}
    third_party/gtest/googletest/include
    third_party/
    third_party/faiss/
    third_party/cxxopts/include
)
link_directories(
    ${CUDA_LIB_PATH}
)

set(SOURCES
    # Utils
    src/util/data_utils.cu
    src/util/debug_utils.cu
    src/util/cuda_utils.cu
    src/util/thrust_utils.cu
    src/util/distance_utils.cu
    src/util/math_utils.cu
    src/util/matrix_broadcast_utils.cu
    src/util/random_utils.cu
    src/util/reduce_utils.cu

    # Kernels
    src/kernels/apply_forces.cu
    src/kernels/attr_forces.cu
    src/kernels/rep_forces.cu
    src/kernels/perplexity_search.cu
    src/kernels/nbodyfft.cu

    # Method files
    src/ext/pymodule_ext.cu
    src/fit_tsne.cu
)


#-------------------------------------------------------------------------------

# Shared Library for Python Binding
#-------------------------------------------------------------------------------
if(BUILD_PYTHON)
    set(PYTHON_SOURCES
        src/python/CHANGES.txt
        src/python/LICENSE.txt
        src/python/MANIFEST.in
        src/python/MANIFEST
        src/python/README.txt
        src/python/setup.py
        src/python/tsnecuda/__init__.py
        src/python/tsnecuda/TSNE.py
        src/python/tsnecuda/test/__init__.py
        src/python/docs/FAQ.txt
    )

    cuda_add_library(tsnecuda SHARED
        ${faiss_gpu_headers} ${faiss_gpu_cpp} ${faiss_gpu_cuh} ${faiss_gpu_cu}
        ${SOURCES}
    )
    set_target_properties(tsnecuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(WITH_ZMQ)
        target_link_libraries(tsnecuda
            gtest
            faiss
            ${CUDA_LIBRARIES}
            ${OpenMP_CXX_FLAGS}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_CUFFT_LIBRARIES}
            ${CUDA_cusparse_LIBRARY}
            #/usr/local/cuda-8.0/lib64/libcudaevrt.a
            ${CUDA_cudadevrt_LIBRARY}
            pthread
            zmq
            ${BLAS_LIB}
            glog
            -Wl,--allow-multiple-definition
            -Wl,-shared
        )
    else()
        target_link_libraries(tsnecuda
            gtest
            faiss
            ${CUDA_LIBRARIES}
            ${OpenMP_CXX_FLAGS}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_CUFFT_LIBRARIES}
            ${CUDA_cusparse_LIBRARY}
            #/usr/local/cuda-8.0/lib64/libcudaevrt.a
            ${CUDA_cudadevrt_LIBRARY}
            pthread
            ${BLAS_LIB}
            glog
            -Wl,--allow-multiple-definition
            -Wl,-shared
        )
    endif()

    # Copy the source files across
    add_custom_target(python_source_files ALL SOURCES ${PYTHON_SOURCES})
    add_dependencies(python_source_files tsnecuda)
    add_custom_command(TARGET python_source_files POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                        ${CMAKE_SOURCE_DIR}/src/python $<TARGET_FILE_DIR:tsnecuda>/python
                    )
    add_custom_command(TARGET python_source_files POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tsnecuda> $<TARGET_FILE_DIR:tsnecuda>/python/tsnecuda)
    add_custom_target(write_version_string_to_python ALL SOURCES ${PYTHON_SOURCES})

    # Write the version string
    add_dependencies(write_version_string_to_python python_source_files)
    add_custom_command(TARGET write_version_string_to_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -DVERSION_STRING=${VERSION_STRING} -DBUILD_NUMBER=${BUILD_NUMBER} -P
        ${CMAKE_CURRENT_BINARY_DIR}/../cmake/write_python_version_string.cmake)
else()
    message("-- Not building python libraries. To build python libraries use -DBUILD_PYTHON=ON")
endif()
#-------------------------------------------------------------------------------

# Test Target
#-------------------------------------------------------------------------------
if(BUILD_TEST)
    cuda_add_executable(tsne_test
        ${faiss_gpu_headers} ${faiss_gpu_cpp} ${faiss_gpu_cuh} ${faiss_gpu_cu}
        ${SOURCES}
        src/test/test.cu
    )
    add_dependencies(tsne_test gtest faiss)
    if(WITH_ZMQ)
        target_link_libraries(tsne_test
            gtest
            gtest_main
            faiss
            ${CUDA_LIBRARIES}
            ${OpenMP_CXX_FLAGS}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_CUFFT_LIBRARIES}
            ${CUDA_cusparse_LIBRARY}
            #/usr/local/cuda-8.0/lib64/libcudaevrt.a
            ${CUDA_cudadevrt_LIBRARY}
            pthread
            zmq
            ${BLAS_LIB}
            glog
            -Wl,--allow-multiple-definition
        )
    else()
        target_link_libraries(tsne_test
            gtest
            gtest_main
            faiss
            ${CUDA_LIBRARIES}
            ${OpenMP_CXX_FLAGS}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_CUFFT_LIBRARIES}
            ${CUDA_cusparse_LIBRARY}
            #/usr/local/cuda-8.0/lib64/libcudaevrt.a
            ${CUDA_cudadevrt_LIBRARY}
            pthread
            ${BLAS_LIB}
            glog
            -Wl,--allow-multiple-definition
        )
endif()
    enable_testing()
    add_test(
        NAME tsne_test
        COMMAND tsne_test
    )
else()
    message("-- Not building tests. To build tests use -DBUILD_TEST=ON")
endif()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Main Executable
#-------------------------------------------------------------------------------
cuda_add_executable(tsne
    ${faiss_gpu_headers} ${faiss_gpu_cpp} ${faiss_gpu_cuh} ${faiss_gpu_cu}
    ${SOURCES}
    src/exe/main.cu
)
add_dependencies(tsne gtest faiss)
if(WITH_ZMQ)
    target_link_libraries(tsne
        gtest
        faiss
        ${CUDA_LIBRARIES}
        ${OpenMP_CXX_FLAGS}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        #/usr/local/cuda-8.0/lib64/libcudaevrt.a
        ${CUDA_cudadevrt_LIBRARY}
        pthread
        zmq
        ${BLAS_LIB}
        glog
        -Wl,--allow-multiple-definition
    )
else()
    target_link_libraries(tsne
        gtest
        faiss
        ${CUDA_LIBRARIES}
        ${OpenMP_CXX_FLAGS}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        #/usr/local/cuda-8.0/lib64/libcudaevrt.a
        ${CUDA_cudadevrt_LIBRARY}
        pthread
        ${BLAS_LIB}
        glog
        -Wl,--allow-multiple-definition
    )
endif()
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
    message("-- Doxygen need to be installed to generate the doxygen documentation")
else (DOXYGEN_FOUND AND BUILD_DOC)
    message("-- Not building documentation. To build documentation use -DBUILD_DOC=ON")
endif (DOXYGEN_FOUND AND BUILD_DOC)
#-------------------------------------------------------------------------------
