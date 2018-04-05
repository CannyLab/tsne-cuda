/**
 * @brief Implementation of different distances
 * 
 * @file distance_utils.cu
 * @author David Chan
 * @date 2018-04-04
 */

 #include "util/distance_utils.h"

 __global__ void assemble_final_result(const float * __restrict__ d_norms_x_2, 
                                       float * __restrict__ d_dots,
                                       const int N)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        const int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i < N) && (j < N)) d_dots[i * N + j] = d_norms_x_2[j] + d_norms_x_2[i] - 2 * d_dots[i * N + j];
    }
// Code from https://github.com/OrangeOwlSolutions/cuBLAS/blob/master/All_pairs_distances.cu
// Expects N x NDIMS matrix in points
void pairwise_dist(cublasHandle_t &handle, 
                   thrust::device_vector<float> &distances, 
                   const thrust::device_vector<float> &points, 
                   const unsigned int N, 
                   const unsigned int NDIMS) 
{
    const unsigned int BLOCKSIZE = 16;
    auto squared_vals = square(points, N * NDIMS);
    auto squared_norms = reduce_sum(handle, squared_vals, N, NDIMS, 1);
    
    float alpha = 1.f;
    float beta = 0.f;
    // Could replace this with cublasSsyrk, might be faster?
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, NDIMS, &alpha,
		                       thrust::raw_pointer_cast(points.data()), N, thrust::raw_pointer_cast(points.data()), N, &beta,
							   thrust::raw_pointer_cast(distances.data()), N));
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(iDivUp(N, BLOCKSIZE), iDivUp(N, BLOCKSIZE));
	assemble_final_result<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(squared_norms.data()), 
                                                 thrust::raw_pointer_cast(distances.data()), N);
                                                 
    distances = sqrt(distances, N * N);
}