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
#include "include/tsne_vars.h"
#include "include/util/cuda_utils.h"

#ifdef __KEPLER__
#define SORT_THREADS 128
#define SORT_BLOCKS 4
#else
#define SORT_THREADS 64
#define SORT_BLOCKS 6
#endif

namespace tsnecuda {
namespace bh {
__global__
__launch_bounds__(SORT_THREADS, SORT_BLOCKS)
void SortKernel(int * __restrict__ cell_sorted, 
                              volatile int * __restrict__ cell_starts, 
                              int * __restrict__ children,
                              const int * __restrict__ cell_counts, 
                              const int num_nodes,
                              const int num_points);

void SortCells(thrust::device_vector<int> &cell_sorted,
                             thrust::device_vector<int> &cell_starts,
                             thrust::device_vector<int> &children,
                             thrust::device_vector<int> &cell_counts,
                             const int num_nodes,
                             const int num_points,
                             const int num_blocks);
}
}

#endif
