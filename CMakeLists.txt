cmake_minimum_required (VERSION 3.20)
project(tsnecuda)
set (CMAKE_PROJECT_VERSION 4)
set (CMAKE_PROJECT_VERSION_MAJOR 4)
set (CMAKE_PROJECT_VERSION_MINOR 0)
set (CMAKE_PROJECT_VERSION_PATCH 0)
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
SET(VERSION_STRING "${CMAKE_PROJECT_VERSION_MAJOR}.${CMAKE_PROJECT_VERSION_MINOR}.${CMAKE_PROJECT_VERSION_PATCH}")
ADD_DEFINITIONS(-DBUILD_NUMBER=${BUILD_NUMBER})
ADD_DEFINITIONS(-DVERSION_STRING=${VERSION_STRING})
# src/exe/main.cu cannot inserting a three-part number (e.g. 3.0.0) in one go without throwing up
ADD_DEFINITIONS(-DVERSION_MAJOR=${CMAKE_PROJECT_VERSION_MAJOR})
ADD_DEFINITIONS(-DVERSION_MINOR=${CMAKE_PROJECT_VERSION_MINOR})
ADD_DEFINITIONS(-DVERSION_PATCH=${CMAKE_PROJECT_VERSION_PATCH})
#-------------------------------------------------------------------------------


# CXX Configuration
#-------------------------------------------------------------------------------
set(OPT_FLAGS
    -std=c++17
    -fopenmp
    -fPIC
    -Wno-sign-compare
    -O3
    -Wall
    -Wextra
    -DFINTEGER=int
)
# x86-only tuning flags; omitted on aarch64 (e.g. Grace/GB200) and other ISAs.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    list(APPEND OPT_FLAGS -m64 -mpopcnt -msse4)
endif()
string(REPLACE ";" " " CXX_FLAGS_STR "${CMAKE_CXX_FLAGS} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS_STR}")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

#-------------------------------------------------------------------------------

# CUDA Configuration
#-------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Architecture selection
# -----------------------------------------------------------------------------
# The kernels use no architecture-specific intrinsics (no warp shuffle/vote,
# no cooperative groups, no tensor-core / wgmma), so a plain `sm_XX` cubin per
# target produces optimal SASS, and a single trailing PTX entry provides JIT
# forward-compatibility for GPUs newer than the toolkit. The `sm_XXa`
# accelerated variants are therefore intentionally omitted.
#
# If a packager or user supplies CMAKE_CUDA_ARCHITECTURES (conda-forge passes it
# through CMAKE_ARGS), honor it and let CMake emit the -gencode flags; we only
# add the non-arch optimization flags. Otherwise pick an optimal ladder for the
# detected toolkit, covering current datacenter GPUs:
#   A100 = sm_80, H100/H200 = sm_90, B200 = sm_100, GB202/RTX50 = sm_120.
if(DEFINED CMAKE_CUDA_ARCHITECTURES
   AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
   AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
    message(STATUS "tsnecuda: honoring CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    set(CUDA_ARCH "")
else()
    # Disable CMake's own arch handling so it does not add conflicting -gencode.
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    if(CUDAToolkit_VERSION_MAJOR EQUAL "13")
        # CUDA 13 removed Maxwell/Pascal/Volta; minimum supported arch is Turing.
        set(CUDA_ARCH
            -gencode=arch=compute_75,code=sm_75
            -gencode=arch=compute_80,code=sm_80
            -gencode=arch=compute_86,code=sm_86
            -gencode=arch=compute_89,code=sm_89
            -gencode=arch=compute_90,code=sm_90
            -gencode=arch=compute_100,code=sm_100
            -gencode=arch=compute_120,code=sm_120
            -gencode=arch=compute_120,code=compute_120
        )
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL "12")
        set(CUDA_ARCH
            -gencode=arch=compute_70,code=sm_70
            -gencode=arch=compute_75,code=sm_75
            -gencode=arch=compute_80,code=sm_80
            -gencode=arch=compute_86,code=sm_86
            -gencode=arch=compute_89,code=sm_89
            -gencode=arch=compute_90,code=sm_90
        )
        # Blackwell (sm_100 / sm_120) requires CUDA >= 12.8.
        if(CUDAToolkit_VERSION_MINOR GREATER_EQUAL "8")
            list(APPEND CUDA_ARCH
                -gencode=arch=compute_100,code=sm_100
                -gencode=arch=compute_120,code=sm_120
                -gencode=arch=compute_120,code=compute_120
            )
        else()
            list(APPEND CUDA_ARCH -gencode=arch=compute_90,code=compute_90)
        endif()
    elseif(CUDAToolkit_VERSION_MAJOR EQUAL "11")
        set(CUDA_ARCH
            -gencode=arch=compute_52,code=sm_52
            -gencode=arch=compute_60,code=sm_60
            -gencode=arch=compute_61,code=sm_61
            -gencode=arch=compute_70,code=sm_70
            -gencode=arch=compute_75,code=sm_75
            -gencode=arch=compute_80,code=sm_80
        )
        # sm_86 available on CUDA >= 11.1; sm_90 only on 11.8+.
        if(CUDAToolkit_VERSION_MINOR GREATER_EQUAL "1")
            list(APPEND CUDA_ARCH -gencode=arch=compute_86,code=sm_86)
        endif()
        list(APPEND CUDA_ARCH -gencode=arch=compute_80,code=compute_80)
    else()
        # Unknown/older toolkit: safe modern default with PTX fallback.
        set(CUDA_ARCH
            -gencode=arch=compute_70,code=sm_70
            -gencode=arch=compute_75,code=sm_75
            -gencode=arch=compute_80,code=sm_80
            -gencode=arch=compute_80,code=compute_80
        )
    endif()
endif()


set(CUDA_OPTS
    -O3
    -Xptxas '-dlcm=cg'
    -Xcompiler '-O3'
    -Xcompiler '-fPIC'
    -Xcompiler '-fopenmp'
    -Xlinker 'muldefs'
)
# x86-only host tuning flags for nvcc's host compiler; omitted on other ISAs.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
    list(APPEND CUDA_OPTS -Xcompiler '-msse4' -Xcompiler '-m64' -Xcompiler '-mpopcnt')
endif()
string (REPLACE ";" " " NVCC_FLAGS_STR "${CUDA_ARCH} ${CUDA_OPTS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS_STR}")
# list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")
#-------------------------------------------------------------------------------

# OpenMP configuration
#-------------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
#-------------------------------------------------------------------------------

# Interactive-visualization (ZMQ) support is intentionally disabled: the ZMQ
# publisher path in fit_tsne is compiled out via -DNO_ZMQ, dropping the zeromq /
# cppzmq build and runtime dependency for a leaner, more portable package.
#-------------------------------------------------------------------------------
ADD_DEFINITIONS(-DNO_ZMQ)
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
    CUDA::cudart
    CUDA::cublas
    # CUDA::cublasLt
    CUDA::cufft
    CUDA::cufftw
    CUDA::cusparse
    OpenMP::OpenMP_CXX
    pthread
    -Wl,--allow-multiple-definition
    ${FAISS_LIBRARIES}
)

# No CPU BLAS/LAPACK is linked: the library's only GEMM is GPU cuBLAS
# (cublasSgemm), and FAISS carries its own BLAS transitively. Dropping the
# find_package(BLAS/LAPACK/MKL) removes a vestigial dependency and, on the
# conda-forge side, the libblas/liblapack pin (incl. the *netlib pin) entirely.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# The standalone C++ CLI (src/exe/main.cu, which pulls in the cxxopts submodule)
# is intentionally not built: the Python module is the supported entry point.
#-------------------------------------------------------------------------------

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
        COMMAND ${CMAKE_COMMAND} -DVERSION_MAJOR=${CMAKE_PROJECT_VERSION_MAJOR} -DVERSION_MINOR=${CMAKE_PROJECT_VERSION_MINOR} -DVERSION_PATCH=${CMAKE_PROJECT_VERSION_PATCH} -DBUILD_NUMBER=${BUILD_NUMBER} -P
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_python_version_string.cmake)
else()
    message("-- Not building python libraries. To build python libraries use -DBUILD_PYTHON=ON")
endif()
