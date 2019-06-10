/**
 * @brief Kernels for computing t-SNE attractive forces with nearest neighbor approximation.
 *
 * @file apply_forces.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */
#ifndef SRC_INCLUDE_KERNELS_BH_ATTR_FORCES_H_
#define SRC_INCLUDE_KERNELS_BH_ATTR_FORCES_H_

#include "../common.h"
#include "../options.h"
#include "../util/cuda_utils.h"

namespace tsnecuda {

 __global__
void ComputePijxQijKernel(
                            float * __restrict__ attr_forces,
                            const float * __restrict__ pij,
                            const float * __restrict__ points,
                            const int * __restrict__ coo_indices,
                            const int num_nodes,
                            const int num_nonzero);

void ComputeAttractiveForces(
                    tsnecuda::GpuOptions &gpu_opt,
                    cusparseHandle_t &handle,
                    cusparseMatDescr_t &descr,
                    thrust::device_vector<float> &attr_forces,
                    thrust::device_vector<float> &sparse_pij,
                    thrust::device_vector<int> &pij_row_ptr,
                    thrust::device_vector<int> &pij_col_ind,
                    thrust::device_vector<int> &coo_indices,
                    thrust::device_vector<float> &points,
                    thrust::device_vector<float> &ones,
                    const int num_nodes,
                    const int num_points,
                    const int num_nonzero);
}

#endif
