// TODO: add copyright

/*
    Compute unnormalized attractive force for barnes-hut approximation of t-SNE.

    Attractive force is given by pij*qij.
*/

#include "kernels/attr_forces.h"

__global__
void ComputePijxQijKernel(
                            float * __restrict__ attr_forces,
                            const float * __restrict__ pij,
                            const float * __restrict__ points,
                            const int * __restrict__ coo_indices,
                            const int num_points,
                            const int num_nonzero)
{
    register int TID, i, j;
    register float ix, iy, jx, jy, dx, dy, pijqij;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_nonzero) return;
    i = coo_indices[2*TID];
    j = coo_indices[2*TID+1];

    ix = points[i]; iy = points[num_points + i];
    jx = points[j]; jy = points[num_points + j];
    dx = ix - jx;
    dy = iy - jy;
    pijqij = pij[TID] / (1 + dx*dx + dy*dy);
    atomicAdd(attr_forces + i, pijqij * dx);
    atomicAdd(attr_forces + num_points + i, pijqij * dy);
}

void tsnecuda::ComputeAttractiveForces(
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
                    const int num_points,
                    const int num_nonzero)
{
    // Computes pij*qij for each i,j
    // TODO: this is bad style
    const int BLOCKSIZE = 1024;
    const int NBLOCKS = iDivUp(num_nonzero, BLOCKSIZE);
    ComputePijxQijKernel<<<NBLOCKS, BLOCKSIZE>>>(
                    thrust::raw_pointer_cast(attr_forces.data()),
                    thrust::raw_pointer_cast(sparse_pij.data()),
                    thrust::raw_pointer_cast(points.data()),
                    thrust::raw_pointer_cast(coo_indices.data()),
                    num_points,
                    num_nonzero);
    GpuErrorCheck(cudaDeviceSynchronize());
}
