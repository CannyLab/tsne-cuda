#ifndef SRC_INCLUDE_KERNELS_REP_FORCES_H_
#define SRC_INCLUDE_KERNELS_REP_FORCES_H_

#include "common.h"
#include "options.h"
#include "util/cuda_utils.h"

namespace tsnecuda {

float ComputeRepulsiveForces(
    thrust::device_vector<float> &repulsive_forces_device,
    thrust::device_vector<float> &normalization_vec_device,
    thrust::device_vector<float> &points_device,
    thrust::device_vector<float> &potentialsQij,
    const int num_points,
    const int n_terms);

void ComputeChargesQij(
    thrust::device_vector<float> &chargesQij,
    thrust::device_vector<float> &points_device,
    const int num_points,
    const int n_terms);
}

#endif
