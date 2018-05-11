/**
 * @brief Kernels for computing bounding box of points in 2 dimensions.
 *
 * @file apply_forces.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */
#ifndef SRC_INCLUDE_KERNELS_BOUNDING_BOX_H_
#define SRC_INCLUDE_KERNELS_BOUNDING_BOX_H_

#include "include/common.h"
#include "include/options.h"
#include "include/tsne_vars.h"
#include "include/util/cuda_utils.h"

namespace tsnecuda {
namespace bh {
__global__
void BoundingBoxKernel(
                       volatile int * __restrict__ cell_starts, 
                       volatile int * __restrict__ children, 
                       volatile float * __restrict__ cell_mass, 
                       volatile float * __restrict__ x_pos_device, 
                       volatile float * __restrict__ y_pos_device, 
                       volatile float * __restrict__ x_max_device, 
                       volatile float * __restrict__ y_max_device, 
                       volatile float * __restrict__ x_min_device, 
                       volatile float * __restrict__ y_min_device,
                       const int num_nodes,
                       const int num_points,
                       const int bounding_box_threads);

void ComputeBoundingBox(tsnecuda::GpuOptions &gpu_opt,
                                thrust::device_vector<int> &cell_starts,
                                thrust::device_vector<int> &children,
                                thrust::device_vector<float> &cell_mass,
                                thrust::device_vector<float> &points,
                                thrust::device_vector<float> &x_max_device,
                                thrust::device_vector<float> &y_max_device,
                                thrust::device_vector<float> &x_min_device,
                                thrust::device_vector<float> &y_min_device,
                                const int num_nodes,
                                const int num_points,
                                const int num_blocks);
}
}

#endif
