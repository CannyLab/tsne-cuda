// TODO: add copyright

/*
    Compute unnormalized attractive force for barnes-hut approximation of t-SNE.

    Attractive force is given by pij*qij.
*/

#include "include/kernels/bh_attr_forces.h"

__global__
void tsnecuda::bh::ComputePijxQijKernel(
                            volatile float * __restrict__ pij_x_qij,
                            const float * __restrict__ pij,
                            const float * __restrict__ points,
                            const int * __restrict__ coo_indices,
                            const int num_nodes,
                            const int num_nonzero)
{
    register int TID, i, j;
    register float ix, iy, jx, jy, dx, dy;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_nonzero) return;
    i = coo_indices[2*TID];
    j = coo_indices[2*TID+1];
    if (i >= num_nodes || i < 0 || j >= num_nodes || j < 0)
        printf("%d, %d\n", i, j);
    ix = points[i]; iy = points[num_nodes + 1 + i];
    jx = points[j]; jy = points[num_nodes + 1 + j];
    dx = ix - jx;
    dy = iy - jy;
    pij_x_qij[TID] = pij[TID] / (1 + dx*dx + dy*dy);
}

void tsnecuda::bh::ComputeAttractiveForces(
                    cusparseHandle_t &handle,
                    cusparseMatDescr_t &descr,
                    thrust::device_vector<float> &attr_forces,
                    thrust::device_vector<float> &pij_x_qij,
                    thrust::device_vector<float> &sparse_pij,
                    thrust::device_vector<int> &pij_row_ptr,
                    thrust::device_vector<int> &pij_col_ind,
                    thrust::device_vector<int> &coo_indices,
                    thrust::device_vector<float> &points,
                    thrust::device_vector<float> &ones,
                    const int num_nodes,
                    const int num_points,
                    const int num_nonzero)
{
    // Computes pij*qij for each i,j
    // TODO: this is bad style
    const int BLOCKSIZE = 1024;
    const int NBLOCKS = iDivUp(num_nonzero, BLOCKSIZE);
    tsnecuda::bh::ComputePijxQijKernel<<<NBLOCKS, BLOCKSIZE>>>(
                    thrust::raw_pointer_cast(pij_x_qij.data()),
                    thrust::raw_pointer_cast(sparse_pij.data()),
                    thrust::raw_pointer_cast(points.data()),
                    thrust::raw_pointer_cast(coo_indices.data()),
                    num_nodes,
                    num_nonzero);
    GpuErrorCheck(cudaDeviceSynchronize());

    // compute forces_i = sum_j pij*qij*normalization*yi
    float alpha = 1.0f;
    float beta = 0.0f;
    CusparseSafeCall(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            num_points, 2, num_points, num_nonzero, &alpha, descr,
                            thrust::raw_pointer_cast(pij_x_qij.data()),
                            thrust::raw_pointer_cast(pij_row_ptr.data()),
                            thrust::raw_pointer_cast(pij_col_ind.data()),
                            thrust::raw_pointer_cast(ones.data()),
                            num_points, &beta, thrust::raw_pointer_cast(attr_forces.data()),
                            num_points));
    GpuErrorCheck(cudaDeviceSynchronize());
    thrust::transform(attr_forces.begin(), attr_forces.begin() + num_points, points.begin(), attr_forces.begin(), thrust::multiplies<float>());
    thrust::transform(attr_forces.begin() + num_points, attr_forces.end(), points.begin() + num_nodes + 1, attr_forces.begin() + num_points, thrust::multiplies<float>());

    // compute forces_i = forces_i - sum_j pij*qij*normalization*yj
    alpha = -1.0f;
    beta = 1.0f;
    CusparseSafeCall(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            num_points, 2, num_points, num_nonzero, &alpha, descr,
                            thrust::raw_pointer_cast(pij_x_qij.data()),
                            thrust::raw_pointer_cast(pij_row_ptr.data()),
                            thrust::raw_pointer_cast(pij_col_ind.data()),
                            thrust::raw_pointer_cast(points.data()),
                            num_nodes + 1, &beta, thrust::raw_pointer_cast(attr_forces.data()),
                            num_points));
    GpuErrorCheck(cudaDeviceSynchronize());
}
