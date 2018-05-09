/**
 * @brief Kernels for computing summary of a quad tree.
 *
 * @file apply_forces.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */
#ifndef SRC_INCLUDE_KERNELS_TREE_SUMMARY_H_
#define SRC_INCLUDE_KERNELS_TREE_SUMMARY_H_

#include "include/common.h"

#ifdef __KEPLER__
#define SUMMARY_THREADS 768
#define SUMMARY_BLOCKS 1
#else
#define SUMMARY_THREADS 128
#define SUMMARY_BLOCKS 6
#endif

namespace tsnecuda {
namespace bh {
__global__
__launch_bounds__(SUMMARY_THREADS, SUMMARY_BLOCKS)
void SummarizationKernel(
                               volatile int * __restrict cell_counts, 
                               volatile float * __restrict cell_mass, 
                               volatile float * __restrict x_pos_device, 
                               volatile float * __restrict y_pos_device,
                               const int * __restrict children,
                               const uint32_t num_nodes,
                               const uint32_t num_points);

void SummarizeTree(thrust::device_vector<int> &cell_counts,
                                 thrust::device_vector<int> &children,
                                 thrust::device_vector<float> &cell_mass,
                                 thrust::device_vector<float> &pts_device,
                                 const uint32_t num_nodes,
                                 const uint32_t num_points,
                                 const uint32_t num_blocks);
}
}

#endif