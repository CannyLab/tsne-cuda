#include "include/kernels/rep_forces.h"


__global__ void compute_repulsive_forces_kernel(
    volatile float * __restrict__ repulsive_forces_device,
    volatile float * __restrict__ normalization_vec_device,
    const float * const xs,
    const float * const ys,
    const float * const potentialsQij,
    const int num_points,
    const int n_terms)
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_points)
        return;

    register float phi1, phi2, phi3, phi4, x_pt, y_pt;

    phi1 = potentialsQij[TID * n_terms + 0];
    phi2 = potentialsQij[TID * n_terms + 1];
    phi3 = potentialsQij[TID * n_terms + 2];
    phi4 = potentialsQij[TID * n_terms + 3];

    x_pt = xs[TID];
    y_pt = ys[TID];

    normalization_vec_device[TID] =
        (1 + x_pt * x_pt + y_pt * y_pt) * phi1 - 2 * (x_pt * phi2 + y_pt * phi3) + phi4;

    repulsive_forces_device[TID] = x_pt * phi1 - phi2;
    repulsive_forces_device[TID + num_points] = y_pt * phi1 - phi3;
}

float tsnecuda::ComputeRepulsiveForces(
    thrust::device_vector<float> &repulsive_forces_device,
    thrust::device_vector<float> &normalization_vec_device,
    thrust::device_vector<float> &points_device,
    thrust::device_vector<float> &potentialsQij,
    const int num_points,
    const int n_terms)
{
    const int num_threads = 1024;
    const int num_blocks = (num_points + num_threads - 1) / num_threads;
    compute_repulsive_forces_kernel<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(repulsive_forces_device.data()),
        thrust::raw_pointer_cast(normalization_vec_device.data()),
        thrust::raw_pointer_cast(points_device.data()),
        thrust::raw_pointer_cast(points_device.data() + num_points),
        thrust::raw_pointer_cast(potentialsQij.data()),
        num_points, n_terms);
    float sumQ = thrust::reduce(
        normalization_vec_device.begin(), normalization_vec_device.end(), 0.0f,
        thrust::plus<float>());
    return sumQ - num_points;
}

__global__ void compute_chargesQij_kernel(
    volatile float * __restrict__ chargesQij,
    const float * const xs,
    const float * const ys,
    const int num_points,
    const int n_terms)
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_points)
        return;

    register float x_pt, y_pt;
    x_pt = xs[TID];
    y_pt = ys[TID];

    chargesQij[TID * n_terms + 0] = 1;
    chargesQij[TID * n_terms + 1] = x_pt;
    chargesQij[TID * n_terms + 2] = y_pt;
    chargesQij[TID * n_terms + 3] = x_pt * x_pt + y_pt * y_pt;
}

void tsnecuda::ComputeChargesQij(
    thrust::device_vector<float> &chargesQij,
    thrust::device_vector<float> &points_device,
    const int num_points,
    const int n_terms)
{
    const int num_threads = 1024;
    const int num_blocks = (num_points + num_threads - 1) / num_threads;
    compute_chargesQij_kernel<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(chargesQij.data()),
        thrust::raw_pointer_cast(points_device.data()),
        thrust::raw_pointer_cast(points_device.data() + num_points),
        num_points, n_terms);
}
