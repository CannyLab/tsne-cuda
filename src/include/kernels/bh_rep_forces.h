/**
 * @brief Kernels for computing t-SNE repulsive forces with barnes hut approximation.
 *
 * @file apply_forces.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */
#ifndef SRC_INCLUDE_KERNELS_BH_REP_FORCES_H_
#define SRC_INCLUDE_KERNELS_BH_REP_FORCES_H_

#include "include/common.h"
#include "include/options.h"
#include "include/util/cuda_utils.h"

// TSNE-Vars
extern __device__ volatile int stepd, bottomd, maxdepthd;
extern __device__ unsigned int blkcntd;
extern __device__ volatile float radiusd;

namespace tsnecuda {
namespace bh {

/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__
void ForceCalculationKernel(volatile int * __restrict__ errd,
                                          volatile float * __restrict__ x_vel_device,
                                          volatile float * __restrict__ y_vel_device,
                                          volatile float * __restrict__ normalization_vec_device,
                                          const int * __restrict__ cell_sorted,
                                          const int * __restrict__ children,
                                          const float * __restrict__ cell_mass,
                                          volatile float * __restrict__ x_pos_device,
                                          volatile float * __restrict__ y_pos_device,
                                          const float theta,
                                          const float epsilon,
                                          const int num_nodes,
                                          const int num_points,
                                          const int maxdepth_bh_tree,
                                          const int repulsive_force_threads);

void ComputeRepulsiveForces(tsnecuda::GpuOptions &gpu_opt,
                                        thrust::device_vector<int> &errd,
                                        thrust::device_vector<float> &repulsive_forces,
                                        thrust::device_vector<float> &normalization_vec,
                                        thrust::device_vector<int> &cell_sorted,
                                        thrust::device_vector<int> &children,
                                        thrust::device_vector<float> &cell_mass,
                                        thrust::device_vector<float> &points,
                                        const float theta,
                                        const float epsilon,
                                        const int num_nodes,
                                        const int num_points,
                                        const int num_blocks);
 
}
}

#endif
