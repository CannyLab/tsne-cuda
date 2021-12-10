cmake_minimum_required (VERSION 3.20)
project(tsnecuda)
set (CMAKE_PROJECT_VERSION 3)
set (CMAKE_PROJECT_VERSION_MAJOR 0)
set (CMAKE_PROJECT_VERSION_MINOR 0)
set (CMAKE_PROJECT_VERSION_PATH 0)
set (CMAKE_SKIP_RULE_DEPENDENCY TRUE)
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
enable_language(CUDA)

# Options
#-------------------------------------------------------------------------------
option(BUILD_PYTHON "Build python if ON" ON)
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
set(OPT_FLAGS
    -std=c++14
    -fopenmp
    -mpopcnt
    -msse4
    -fPIC
    -m64
    -Wno-sign-compare
    -g
    -O3
    -Wall
    -Wextra
    -DFINTEGER=int
)
string(REPLACE ";" " " CXX_FLAGS_STR "${CMAKE_CXX_FLAGS} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS_STR}")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#-------------------------------------------------------------------------------

# CUDA Configuration
#-------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(CUDAToolkit_VERSION_MAJOR EQUAL "10")
    set(CUDA_ARCH
        -gencode=arch=compute_30,code=sm_30
        -gencode=arch=compute_35,code=sm_35
        -gencode=arch=compute_37,code=sm_37
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_60,code=sm_60
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_70,code=sm_70
        -gencode=arch=compute_75,code=sm_75
    )
elseif(CUDAToolkit_VERSION_MAJOR EQUAL "11" AND CUDAToolkit_VERSION_MINOR LESS "1")
    set(CUDA_ARCH
        -gencode=arch=compute_35,code=sm_35
        -gencode=arch=compute_37,code=sm_37
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_60,code=sm_60
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_70,code=sm_70
        -gencode=arch=compute_75,code=sm_75
        -gencode=arch=compute_80,code=sm_80
    )
elseif(CUDAToolkit_VERSION_MAJOR EQUAL "11")
    set(CUDA_ARCH
        -gencode=arch=compute_35,code=sm_35
        -gencode=arch=compute_37,code=sm_37
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_60,code=sm_60
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_70,code=sm_70
        -gencode=arch=compute_75,code=sm_75
        -gencode=arch=compute_80,code=sm_80
        -gencode=arch=compute_86,code=sm_86
    )
else()
    set(CUDA_ARCH
        -gencode=arch=compute_30,code=sm_30
        -gencode=arch=compute_35,code=sm_35
        -gencode=arch=compute_37,code=sm_37
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_60,code=sm_60
        -gencode=arch=compute_61,code=sm_61
        )
endif()


set(CUDA_OPTS
    -O3
    -g
    -Xptxas '-dlcm=cg'
    -Xcompiler '-O3'
    -Xcompiler '-fPIC'
    -Xcompiler '-fopenmp'
    -Xcompiler '-msse4'
    -Xcompiler '-m64'
    -Xcompiler '-mpopcnt'
    -Xcompiler '-g'
    -Xlinker 'muldefs'
)
string (REPLACE ";" " " NVCC_FLAGS_STR "${CUDA_ARCH} ${CUDA_OPTS}")
set(CMAKE_CUDA_FLAGS "${NVCC_FLAGS_STR}")
# list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")
#-------------------------------------------------------------------------------

# OpenMP configuration
#-------------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
#-------------------------------------------------------------------------------

# GTEST Configuration
#-------------------------------------------------------------------------------
find_package(gflags REQUIRED)
# find_package(GLog REQUIRED)
find_package(GTest REQUIRED)
#-------------------------------------------------------------------------------

# ZMQ Configuration
#-------------------------------------------------------------------------------
find_package(ZMQ)
if(NOT ${ZMQ_FOUND})
    ADD_DEFINITIONS(-DNO_ZMQ)
    set(ZMQ_INCLUDE_DIR "")
    set(ZMQ_LIBRARIES "")
    message("-- Not building with ZMQ. Interactive visualization disabled. To build with ZMQ use -DWITH_ZMQ=ON")
endif()
#-------------------------------------------------------------------------------

# FAISS Configuration
#-------------------------------------------------------------------------------
find_package(FAISS REQUIRED)
if(NOT ${GPU_FAISS_FOUND})
    message("-- GPU FAISS not installed. Please install FAISS for GPU.")
endif()
include_directories(${FAISS_INCLUDE_DIR})

# Project Setup
#-------------------------------------------------------------------------------
include_directories(
    src/
    src/include
    ${CUDA_INCLUDE_DIRS}
    third_party/
    third_party/cxxopts/include/
    ${ZMQ_INCLUDE_DIR}
)
link_directories(
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
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

add_library(tsnecuda SHARED ${SOURCES})
# set_property(TARGET tsnecuda PROPERTY POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(tsnecuda PRIVATE
    gflags
    gtest
    CUDA::cudart
    CUDA::cublas
    # CUDA::cublasLt
    CUDA::cufft
    CUDA::cufftw
    CUDA::cusparse
    OpenMP::OpenMP_CXX
    pthread
    -Wl,--allow-multiple-definition
    ${ZMQ_LIBRARIES}
    ${FAISS_LIBRARIES}
)

# BLAS configuration
#-------------------------------------------------------------------------------
find_package(MKL)
if(MKL_FOUND)
  target_link_libraries(tsnecuda PRIVATE ${MKL_LIBRARIES})
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(tsnecuda PRIVATE ${BLAS_LIBRARIES})

  find_package(LAPACK REQUIRED)
  target_link_libraries(tsnecuda PRIVATE ${LAPACK_LIBRARIES})
endif()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Main executable
#-------------------------------------------------------------------------------
add_executable(tsne src/exe/main.cu)
target_link_libraries(tsne tsnecuda)

#-------------------------------------------------------------------------------
# Python library copy and build
#-------------------------------------------------------------------------------
if(BUILD_PYTHON)
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
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
