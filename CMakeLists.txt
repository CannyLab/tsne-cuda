cmake_minimum_required (VERSION 2.8)
project(cuda-tsne)
set (cuda-tsne_VERSION_MAJOR 0)
set (cuda-tsne_VERSION_MINOR 1)


# CUDA Configuration
#-------------------------------------------------------------------------------
find_package(CUDA REQUIRED)
#-------------------------------------------------------------------------------
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=compute_61 -code=sm_61 -ccbin g++ -std=c++11 -g)
#-------------------------------------------------------------------------------

include_directories(
    src/include
    ${CUDA_INCLUDE_DIRECTORIES}
)

cuda_add_executable(
    tsne_utils
    src/tsne_utils.cu
    src/Utilities.cu)
target_link_libraries(tsne_utils ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
