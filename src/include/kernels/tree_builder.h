/**
 * @brief Kernels for computing t-SNE attractive forces with nearest neighbor approximation.
 *
 * @file apply_forces.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */
#ifndef SRC_INCLUDE_KERNELS_TREE_BUILDER_H_
#define SRC_INCLUDE_KERNELS_TREE_BUILDER_H_

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
void ClearKernel1(volatile int * __restrict__ children, const int num_nodes, const int num_points);

__global__
void TreeBuildingKernel(volatile int * __restrict__ errd, 
                                        volatile int * __restrict__ children, 
                                        volatile float * __restrict__ x_pos_device, 
                                        volatile float * __restrict__ y_pos_device,
                                        const int num_nodes,
                                        const int num_points);

__global__
void ClearKernel2(volatile int * __restrict__ cell_starts, volatile float * __restrict__ cell_mass, const int num_nodes);

void BuildTree(tsnecuda::GpuOptions &gpu_opt,
                    thrust::device_vector<int> &errd,
                    thrust::device_vector<int> &children,
                    thrust::device_vector<int> &cell_starts,
                    thrust::device_vector<float> &cell_mass,
                    thrust::device_vector<float> &points,
                    const int num_nodes,
                    const int num_points,
                    const int num_blocks);
}
}

#endif
