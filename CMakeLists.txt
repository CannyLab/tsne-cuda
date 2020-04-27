cmake_minimum_required (VERSION 3.10)
project(tsnecuda)
set (CMAKE_PROJECT_VERSION 2)
set (CMAKE_PROJECT_VERSION_MAJOR 2)
set (CMAKE_PROJECT_VERSION_MINOR 2)
set (CMAKE_PROJECT_VERSION_PATH 0)
set (CMAKE_SKIP_RULE_DEPENDENCY TRUE)
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
enable_language(CUDA)

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
set(OPT_FLAGS
    -std=c++11
    -fopenmp
    -mpopcnt
    -msse4
    # -fPIC
    -m64
    -Wno-sign-compare
    -g
    -ggdb
    -O0
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
set(CMAKE_CUDA_STANDARD 11)
set(CUDA_ARCH
    -gencode=arch=compute_30,code=sm_30
    -gencode=arch=compute_35,code=sm_35
    # -gencode=arch=compute_50,code=sm_50
    -gencode=arch=compute_52,code=sm_52
    # -gencode=arch=compute_60,code=sm_60
    -gencode=arch=compute_61,code=sm_61
    # -gencode=arch=compute_70,code=sm_70
)
set(CUDA_OPTS
    -O0
    -g
    -ggdb
    -Xptxas '-dlcm=cg'
    -std=c++11
    -Xcompiler '-O0'
    # -Xcompiler '-fPIC'
    -Xcompiler '-fopenmp'
    -Xcompiler '-msse4'
    -Xcompiler '-m64'
    -Xcompiler '-mpopcnt'
    -Xcompiler '-g'
    -Xcompiler '-ggdb'
    -Xlinker 'muldefs'
    # -fPIC
)
string (REPLACE ";" " " NVCC_FLAGS_STR "${CUDA_ARCH} ${CUDA_OPTS}")
set(CMAKE_CUDA_FLAGS "${NVCC_FLAGS_STR}")
# list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")
#-------------------------------------------------------------------------------

# OpenMP configuration
#-------------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
#-------------------------------------------------------------------------------

# BLAS configuration
#-------------------------------------------------------------------------------
find_package(BLAS REQUIRED)
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
set(FAISS_HEADERS_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/
)
file(GLOB FAISS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/impl/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/utils/*.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/gpu/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/gpu/impl/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/gpu/utils/*.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/gpu/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/gpu/impl/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/gpu/utils/*.cu


    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/gpu/utils/blockselect/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/gpu/utils/warpselect/*.cu
)

# Add libraries
add_library(gpufaiss STATIC ${FAISS_SRC})
# set_property(TARGET gpufaiss PROPERTY POSITION_INDEPENDENT_CODE TRUE)
target_include_directories(gpufaiss PRIVATE ${FAISS_HEADERS_DIRS})
target_link_libraries(gpufaiss
    ${BLAS_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
)

# Project Setup
#-------------------------------------------------------------------------------
include_directories(
    src/
    src/include
    ${CUDAToolkit_INCLUDE_DIRS}
    third_party/gtest/googletest/include/
    third_party/
    third_party/cxxopts/include/
)
link_directories(
    ${CUDAToolkit_LIBRARY_DIR}
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

add_library(tsnecuda STATIC ${SOURCES})
# set_property(TARGET tsnecuda PROPERTY POSITION_INDEPENDENT_CODE TRUE)
target_link_libraries(tsnecuda
    gtest
    glog
    gpufaiss
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
    CUDA::cufftw
    CUDA::cusparse
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    pthread
    -Wl,--allow-multiple-definition
)

add_executable(tsne src/exe/main.cu)
target_link_libraries(tsne tsnecuda)

#-------------------------------------------------------------------------------

# Shared Library for Python Binding
#-------------------------------------------------------------------------------
# if(BUILD_PYTHON)
#     set(PYTHON_SOURCES
#         src/python/CHANGES.txt
#         src/python/LICENSE.txt
#         src/python/MANIFEST.in
#         src/python/MANIFEST
#         src/python/README.txt
#         src/python/setup.py
#         src/python/tsnecuda/__init__.py
#         src/python/tsnecuda/TSNE.py
#         src/python/tsnecuda/test/__init__.py
#         src/python/docs/FAQ.txt
#     )

#     cuda_add_library(tsnecuda SHARED
#         ${SOURCES}
#     )
#     set_target_properties(tsnecuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
#     if(WITH_ZMQ)
#         target_link_libraries(tsnecuda
#             gtest
#             libgpufaiss
#             ${CUDA_LIBRARIES}
#             ${OpenMP_CXX_FLAGS}
#             ${CUDA_CUBLAS_LIBRARIES}
#             ${CUDA_CUFFT_LIBRARIES}
#             ${CUDA_cusparse_LIBRARY}
#             #/usr/local/cuda-8.0/lib64/libcudaevrt.a
#             ${CUDA_cudadevrt_LIBRARY}
#             pthread
#             zmq
#             ${BLAS_LIB}
#             glog
#             -Wl,--allow-multiple-definition
#             -Wl,-shared
#         )
#     else()
#         target_link_libraries(tsnecuda
#             gtest
#             libgpufaiss
#             ${CUDA_LIBRARIES}
#             ${OpenMP_CXX_FLAGS}
#             ${CUDA_CUBLAS_LIBRARIES}
#             ${CUDA_CUFFT_LIBRARIES}
#             ${CUDA_cusparse_LIBRARY}
#             #/usr/local/cuda-8.0/lib64/libcudaevrt.a
#             ${CUDA_cudadevrt_LIBRARY}
#             pthread
#             ${BLAS_LIB}
#             glog
#             -Wl,--allow-multiple-definition
#             -Wl,-shared
#         )
#     endif()

#     # Copy the source files across
#     add_custom_target(python_source_files ALL SOURCES ${PYTHON_SOURCES})
#     add_dependencies(python_source_files tsnecuda)
#     add_custom_command(TARGET python_source_files POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_directory
#                         ${CMAKE_SOURCE_DIR}/src/python $<TARGET_FILE_DIR:tsnecuda>/python
#                     )
#     add_custom_command(TARGET python_source_files POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tsnecuda> $<TARGET_FILE_DIR:tsnecuda>/python/tsnecuda)
#     add_custom_target(write_version_string_to_python ALL SOURCES ${PYTHON_SOURCES})

#     # Write the version string
#     add_dependencies(write_version_string_to_python python_source_files)
#     add_custom_command(TARGET write_version_string_to_python POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -DVERSION_STRING=${VERSION_STRING} -DBUILD_NUMBER=${BUILD_NUMBER} -P
#         ${CMAKE_CURRENT_BINARY_DIR}/../cmake/write_python_version_string.cmake)
# else()
#     message("-- Not building python libraries. To build python libraries use -DBUILD_PYTHON=ON")
# endif()
# #-------------------------------------------------------------------------------

# # Test Target
# #-------------------------------------------------------------------------------
# if(BUILD_TEST)
#     cuda_add_executable(tsne_test
#         ${SOURCES}
#         src/test/test.cu
#     )
#     add_dependencies(tsne_test gtest faiss_core)
#     if(WITH_ZMQ)
#         target_link_libraries(tsne_test
#             gtest
#             gtest_main
#             libgpufaiss
#             ${CUDA_LIBRARIES}
#             ${OpenMP_CXX_FLAGS}
#             ${CUDA_CUBLAS_LIBRARIES}
#             ${CUDA_CUFFT_LIBRARIES}
#             ${CUDA_cusparse_LIBRARY}
#             #/usr/local/cuda-8.0/lib64/libcudaevrt.a
#             ${CUDA_cudadevrt_LIBRARY}
#             pthread
#             zmq
#             ${BLAS_LIB}
#             glog
#             -Wl,--allow-multiple-definition
#         )
#     else()
#         target_link_libraries(tsne_test
#             gtest
#             gtest_main
#             faiss
#             ${CUDA_LIBRARIES}
#             ${OpenMP_CXX_FLAGS}
#             ${CUDA_CUBLAS_LIBRARIES}
#             ${CUDA_CUFFT_LIBRARIES}
#             ${CUDA_cusparse_LIBRARY}
#             #/usr/local/cuda-8.0/lib64/libcudaevrt.a
#             ${CUDA_cudadevrt_LIBRARY}
#             pthread
#             ${BLAS_LIB}
#             glog
#             -Wl,--allow-multiple-definition
#         )
# endif()
#     enable_testing()
#     add_test(
#         NAME tsne_test
#         COMMAND tsne_test
#     )
# else()
#     message("-- Not building tests. To build tests use -DBUILD_TEST=ON")
# endif()
# #-------------------------------------------------------------------------------

# #-------------------------------------------------------------------------------
# # Main Executable
# #-------------------------------------------------------------------------------
# cuda_add_executable(tsne
#     ${SOURCES}
#     src/exe/main.cu
# )
# add_dependencies(tsne gtest faiss_core)
# if(WITH_ZMQ)
#     target_link_libraries(tsne
#         gtest
#         libgpufaiss
#         ${CUDA_LIBRARIES}
#         ${OpenMP_CXX_FLAGS}
#         ${CUDA_CUBLAS_LIBRARIES}
#         ${CUDA_CUFFT_LIBRARIES}
#         ${CUDA_cusparse_LIBRARY}
#         #/usr/local/cuda-8.0/lib64/libcudaevrt.a
#         ${CUDA_cudadevrt_LIBRARY}
#         pthread
#         zmq
#         ${BLAS_LIB}
#         glog
#         -Wl,--allow-multiple-definition
#     )
# else()
#     target_link_libraries(tsne
#         gtest
#         libgpufaiss
#         ${CUDA_LIBRARIES}
#         ${OpenMP_CXX_FLAGS}
#         ${CUDA_CUBLAS_LIBRARIES}
#         ${CUDA_CUFFT_LIBRARIES}
#         ${CUDA_cusparse_LIBRARY}
#         #/usr/local/cuda-8.0/lib64/libcudaevrt.a
#         ${CUDA_cudadevrt_LIBRARY}
#         pthread
#         ${BLAS_LIB}
#         glog
#         -Wl,--allow-multiple-definition
#     )
# endif()
# #-------------------------------------------------------------------------------

# # Documentation
# #-------------------------------------------------------------------------------
# # first we can indicate the documentation build as an option and set it to ON by default
# option(BUILD_DOC "Build documentation" OFF)

# # check if Doxygen is installed
# find_package(Doxygen)
# if (DOXYGEN_FOUND AND BUILD_DOC)
#     # set input and output files
#     set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
#     set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#     # request to configure the file
#     configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
#     # note the option ALL which allows to build the docs together with the application
#     add_custom_target( doc_doxygen ALL
#         COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#         COMMENT "Generating API documentation with Doxygen"
#         VERBATIM )
# elseif(NOT DOXYGEN_FOUND AND BUILD_DOC)
#     message("-- Doxygen need to be installed to generate the doxygen documentation")
# else (DOXYGEN_FOUND AND BUILD_DOC)
#     message("-- Not building documentation. To build documentation use -DBUILD_DOC=ON")
# endif (DOXYGEN_FOUND AND BUILD_DOC)
# #-------------------------------------------------------------------------------
