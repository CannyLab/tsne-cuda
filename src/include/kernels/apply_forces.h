/**
 * @brief Kernels for applying t-SNE forces with momentum.
 *
 * @file apply_forces.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_KERNELS_APPLY_FORCES_H_
#define SRC_INCLUDE_KERNELS_APPLY_FORCES_H_

#include "include/common.h"
#include "include/util/cuda_utils.h"

#ifdef __KEPLER__
#define INTEGRATION_THREADS 1024
#define INTEGRATION_BLOCKS 2
#else
#define INTEGRATION_THREADS 1024
#define INTEGRATION_BLOCKS 1
#endif
 
namespace tsnecuda {
namespace bh {
__global__
__launch_bounds__(INTEGRATION_THREADS, INTEGRATION_BLOCKS)
void IntegrationKernel(
                                 volatile float * __restrict__ points,
                                 volatile float * __restrict__ attr_forces,
                                 volatile float * __restrict__ rep_forces,
                                 volatile float * __restrict__ gains,
                                 volatile float * __restrict__ old_forces,
                                 const float eta,
                                 const float normalization,
                                 const float momentum,
                                 const float exaggeration,
                                 const uint32_t num_nodes,
                                 const uint32_t num_points
                      );

void ApplyForces(thrust::device_vector<float> &points,
                               thrust::device_vector<float> &attr_forces,
                               thrust::device_vector<float> &rep_forces,
                               thrust::device_vector<float> &gains, 
                               thrust::device_vector<float> &old_forces,
                               const float eta,
                               const float normalization,
                               const float momentum,
                               const float exaggeration,
                               const uint32_t num_nodes,
                               const uint32_t num_points,
                               const uint32_t num_blocks
                      );
}

namespace naive {

__global__
__launch_bounds__(INTEGRATION_THREADS, INTEGRATION_BLOCKS)
void IntegrationKernel(
                                 volatile float * __restrict__ points,
                                 volatile float * __restrict__ forces,
                                 volatile float * __restrict__ gains,
                                 volatile float * __restrict__ old_forces,
                                 const float eta,
                                 const float momentum,
                                 const uint32_t num_points
                      );


void ApplyForces(thrust::device_vector<float> &points,
                                  thrust::device_vector<float> &forces,
                                  thrust::device_vector<float> &gains,
                                  thrust::device_vector<float> &old_forces,
                                  const float eta,
                                  const float momentum,
                                  const uint32_t num_points,
                                  const uint32_t num_blocks);

}
}
#endif
