/**
 * @brief Kernels for sorting tree cells by morton code.
 *
 * @file tree_sort.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */
#ifndef SRC_INCLUDE_KERNELS_TREE_SORT_H_
#define SRC_INCLUDE_KERNELS_TREE_SORT_H_

#include "include/common.h"
#include "include/options.h"
#include "include/util/cuda_utils.h"

//TSNE-Vars
extern __device__ volatile int stepd, bottomd, maxdepthd;
extern __device__ unsigned int blkcntd;
extern __device__ volatile float radiusd;

namespace tsnecuda {
namespace bh {
__global__
void SortKernel(int * __restrict__ cell_sorted, 
                              volatile int * __restrict__ cell_starts, 
                              int * __restrict__ children,
                              const int * __restrict__ cell_counts, 
                              const int num_nodes,
                              const int num_points);

void SortCells(tsnecuda::GpuOptions &gpu_opt,
                thrust::device_vector<int> &cell_sorted,
                thrust::device_vector<int> &cell_starts,
                thrust::device_vector<int> &children,
                thrust::device_vector<int> &cell_counts,
                const int num_nodes,
                const int num_points,
                const int num_blocks);
}
}

#endif
