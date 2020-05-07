// TODO: add copyright

/*
    Compute unnormalized attractive force for barnes-hut approximation of t-SNE.

    Attractive force is given by pij*qij.
*/

#include "kernels/attr_forces.h"

__global__ void ComputePijxQijKernel(
    float *__restrict__ attr_forces,
    const float *__restrict__ pij,
    const float *__restrict__ points,
    const int *__restrict__ coo_indices, // Row vector of 2x NNz which has COO format
    const int num_points,
    const int num_nonzero)
{
    register int TID, i, j;
    register float ix, iy, jx, jy, dx, dy, pijqij;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_nonzero)
        return;

    i = coo_indices[2 * TID];     // Point A
    j = coo_indices[2 * TID + 1]; // Point B

    ix = points[i];                                       // Ax
    iy = points[num_points + i];                          //Ay
    jx = points[j];                                       //Bx
    jy = points[num_points + j];                          // By
    dx = ix - jx;                                         // X distance
    dy = iy - jy;                                         // Y distance
    pijqij = pij[TID] / (1 + dx * dx + dy * dy);          // Normalizing factor
    atomicAdd(attr_forces + i, pijqij * dx);              // Update with X distances
    atomicAdd(attr_forces + num_points + i, pijqij * dy); // Update with Y distances
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

__global__ void ComputePijxQijKernelV2(
    float *__restrict__ attr_forces,
    const float *__restrict__ points,
    const float *__restrict__ pij,
    const int *__restrict__ pij_row_ptr,
    const int *__restrict__ pij_col_ind,
    const int num_points)
{
    register int TID, i, j, jidx, jidx_end;
    register float ix, iy, jx, jy, dx, dy, pijqij;
    register float acc_x = 0, acc_y = 0;
    TID = threadIdx.x + blockIdx.x * blockDim.x;

    if (TID >= num_points)
        return;

    // Thread ID is point set
    i = TID;
    ix = points[i];
    iy = points[num_points + i];
    jidx_end = pij_row_ptr[TID + 1];
    for (jidx = pij_row_ptr[TID]; jidx < jidx_end; jidx++)
    {
        j = pij_col_ind[jidx];
        jx = points[j];
        jy = points[num_points + j];
        dx = ix - jx; // X distance
        dy = iy - jy; // Y distance
        pijqij = pij[jidx] / (1 + dx * dx + dy * dy);
        acc_x += pijqij * dx;
        acc_y += pijqij * dy;
    }
    attr_forces[i] = acc_x;
    attr_forces[num_points + i] = acc_y;
}

void tsnecuda::ComputeAttractiveForcesV2(
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
    const int BLOCKSIZE = 1024;
    const int NBLOCKS = iDivUp(num_points, BLOCKSIZE);
    ComputePijxQijKernelV2<<<NBLOCKS, BLOCKSIZE>>>(
        thrust::raw_pointer_cast(attr_forces.data()),
        thrust::raw_pointer_cast(points.data()),
        thrust::raw_pointer_cast(sparse_pij.data()),
        thrust::raw_pointer_cast(pij_row_ptr.data()),
        thrust::raw_pointer_cast(pij_col_ind.data()),
        num_points);
    GpuErrorCheck(cudaDeviceSynchronize());
}

__global__ void ComputePijxQijKernelV3(
    // float *__restrict__ attr_forces,
    float *__restrict__ workspace_x,
    float *__restrict__ workspace_y,
    const float *__restrict__ pij,
    const int *__restrict__ pij_ind,
    const float *__restrict__ points,
    const int num_points,
    const int num_neighbors)
{
    register int TID, i, j;
    register float ix, iy, jx, jy, dx, dy, pijqij;
    TID = threadIdx.x + blockIdx.x * blockDim.x; // This is the location in the pij matrix
    if (TID >= num_points * num_neighbors)
        return;

    i = TID / num_neighbors;
    j = pij_ind[TID];

    ix = points[i];
    iy = points[num_points + i];
    jx = points[j];
    jy = points[num_points + j];
    dx = ix - jx; // X distance
    dy = iy - jy; // Y distance
    pijqij = pij[TID] / (1 + dx * dx + dy * dy);

    // Convert to atomics
    // atomicAdd(attr_forces + i, pijqij * dx);              // Update with X distances
    // atomicAdd(attr_forces + num_points + i, pijqij * dy); // Update with Y distances

    workspace_x[TID] = pijqij * dx;
    workspace_y[TID] = pijqij * dy;
}

__global__ void reduce_sum_kernel(
    float *__restrict__ attr_forces,
    const float *__restrict__ workspace_x,
    const float *__restrict__ workspace_y,
    const int num_points,
    const int num_neighbors)
{
    register int TID, jend, j;
    register float acc_x, acc_y;
    TID = threadIdx.x + blockIdx.x * blockDim.x; // This is the location in the pij matrix
    if (TID >= num_points)
        return;

    acc_x = 0.0f;
    acc_y = 0.0f;
    jend = (TID + 1) * num_neighbors;
    for (j = TID * num_neighbors; j < jend; j++)
    {
        acc_x += workspace_x[j];
        acc_y += workspace_y[j];
    }

    attr_forces[TID] = acc_x;
    attr_forces[num_points + TID] = acc_y;
}

void tsnecuda::ComputeAttractiveForcesV3(
    cublasHandle_t &handle,
    tsnecuda::GpuOptions &gpu_opt,
    thrust::device_vector<float> &attr_forces,
    thrust::device_vector<float> &pij_device,
    thrust::device_vector<int> &pij_indices_device,
    thrust::device_vector<float> &pij_workspace_device,
    thrust::device_vector<float> &points_device,
    thrust::device_vector<float> &ones_vec,
    const int num_points,
    const int num_neighbors)
{
    // Step 1: Store the independent pij values  for x and y in the workspace
    thrust::fill(pij_workspace_device.begin(), pij_workspace_device.end(), 0.0f);
    const int BLOCKSIZE = 1024;
    const int NBLOCKS = iDivUp(num_points * num_neighbors, BLOCKSIZE);
    ComputePijxQijKernelV3<<<NBLOCKS, BLOCKSIZE>>>(
        // thrust::raw_pointer_cast(attr_forces.data()),
        thrust::raw_pointer_cast(pij_workspace_device.data()),                              // Workspace X
        thrust::raw_pointer_cast(pij_workspace_device.data()) + num_points * num_neighbors, // Workspace Y
        thrust::raw_pointer_cast(pij_device.data()),                                        // pij
        thrust::raw_pointer_cast(pij_indices_device.data()),                                // pij_indices
        thrust::raw_pointer_cast(points_device.data()),                                     // points
        num_points,
        num_neighbors);

    GpuErrorCheck(cudaDeviceSynchronize());

    const int NBLOCKS2 = iDivUp(num_points, BLOCKSIZE);
    reduce_sum_kernel<<<NBLOCKS2, BLOCKSIZE>>>(
        thrust::raw_pointer_cast(attr_forces.data()),
        thrust::raw_pointer_cast(pij_workspace_device.data()),                              // Workspace X
        thrust::raw_pointer_cast(pij_workspace_device.data()) + num_points * num_neighbors, // Workspace Y
        num_points,
        num_neighbors);

    GpuErrorCheck(cudaDeviceSynchronize());

    // // Setp 2: Reduce the X pij values into the attractive forces
    // float kAlpha = 1.0f;
    // float kBeta = 0.0f;
    // CublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, num_points, num_neighbors, &kAlpha,
    //                            thrust::raw_pointer_cast(pij_workspace_device.data()), num_points,
    //                            thrust::raw_pointer_cast(ones_vec.data()), 1, &kBeta,
    //                            thrust::raw_pointer_cast(attr_forces.data()), 1));

    // // Setp 2: Reduce the Y pij values into the attractive forces
    // CublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, num_points, num_neighbors, &kAlpha,
    //                            thrust::raw_pointer_cast(pij_workspace_device.data()) + num_points * num_neighbors, num_points,
    //                            thrust::raw_pointer_cast(ones_vec.data()), 1, &kBeta,
    //                            thrust::raw_pointer_cast(attr_forces.data()) + num_points, 1));
    // GpuErrorCheck(cudaDeviceSynchronize());
}
