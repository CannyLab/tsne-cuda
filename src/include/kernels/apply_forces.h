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

#include "../common.h"
#include "../options.h"
#include "../util/cuda_utils.h"

namespace tsnecuda {
__global__
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
                                 const int num_nodes,
                                 const int num_points
                      );

void ApplyForces(
                    tsnecuda::GpuOptions &gpu_opt,
                    thrust::device_vector<float> &points,
                    thrust::device_vector<float> &attr_forces,
                    thrust::device_vector<float> &rep_forces,
                    thrust::device_vector<float> &gains,
                    thrust::device_vector<float> &old_forces,
                    const float eta,
                    const float normalization,
                    const float momentum,
                    const float exaggeration,
                    const int num_nodes,
                    const int num_points,
                    const int num_blocks
            );
}
#endif
