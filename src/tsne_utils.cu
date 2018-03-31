#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <random>
#include <sys/time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <stdexcept>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include "tsne_utils.cuh"
#include "Utilities.cuh"

struct func_square {
	__host__ __device__ double operator()(const float &x) const { return x * x; }
};

struct func_sqrt {
    __host__ __device__ double operator()(const float &x) const { return pow(x, 0.5); }
};

__global__ void assemble_final_result(const float * __restrict__ d_norms_x_2, float * __restrict__ d_dots,
									  const int N) {

	const int i = threadIdx.x + blockIdx.x * gridDim.x;
	const int j = threadIdx.y + blockIdx.y * gridDim.y;

	if ((i < N) && (j < N)) d_dots[i * N + j] = d_norms_x_2[j] + d_norms_x_2[i] - 2 * d_dots[i * N + j];
    
}

// Performs the operation matrix[i,:] = matrix[i,:] - alpha*vector for each row 0 <= i < N
// This is in place addition
__global__ void _add_row_vec(float * __restrict__ matrix, const float * __restrict__ vector, const unsigned int N, const unsigned int M, const float alpha) {
    const unsigned int TID = threadIdx.x + blockIdx.x * gridDim.x;
    const unsigned int i = TID / M;
    const unsigned int j = TID % M;

    if (i < N) matrix[i * M + j] = matrix[i * M + j] + alpha*vector[j];
}

void add_row_vec(thrust::device_vector<float> matrix, thrust::device_vector<float> vector, const unsigned int N, const unsigned int M, const float alpha) {
    const unsigned int BLOCKSIZE = 32;
    const unsigned int NBLOCKS = iDivUp(N * M, BLOCKSIZE);
    _add_row_vec<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(matrix.data()), thrust::raw_pointer_cast(vector.data()), N, M, alpha);
}

// Performs the operation matrix[i,:] = alpha*matrix[i,:]*vector for each row 0 <= i < N
__global__ void _mul_row_vec(float * __restrict__ matrix, const float * __restrict__ vector, const unsigned int N, const unsigned int M, const float alpha) {
    const unsigned int TID = threadIdx.x + blockIdx.x * gridDim.x;
    const unsigned int i = TID / M;
    const unsigned int j = TID % M;

    if (i < N) matrix[i * M + j] = alpha*matrix[i * M + j]*vector[j];
}

void mul_row_vec(thrust::device_vector<float> matrix, thrust::device_vector<float> vector, const unsigned int N, const unsigned int M, const float alpha) {
    const unsigned int BLOCKSIZE = 32;
    const unsigned int NBLOCKS = iDivUp(N * M, BLOCKSIZE);
    _mul_row_vec<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(matrix.data()), thrust::raw_pointer_cast(vector.data()), N, M, alpha);
}

// Performs the operation matrix[i,:] = alpha*matrix[i,:]/vector for each row 0 <= i < N
__global__ void _div_row_vec(float * __restrict__ matrix, const float * __restrict__ vector, const unsigned int N, const unsigned int M, const float alpha) {
    const unsigned int TID = threadIdx.x + blockIdx.x * gridDim.x;
    const unsigned int i = TID / M;
    const unsigned int j = TID % M;

    if (i < N) matrix[i * M + j] = alpha*matrix[i * M + j]/vector[j];
}

void div_row_vec(thrust::device_vector<float> matrix, thrust::device_vector<float> vector, const unsigned int N, const unsigned int M, const float alpha) {
    const unsigned int BLOCKSIZE = 32;
    const unsigned int NBLOCKS = iDivUp(N * M, BLOCKSIZE);
    _div_row_vec<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(matrix.data()), thrust::raw_pointer_cast(vector.data()), N, M, alpha);
}

// Code from https://github.com/OrangeOwlSolutions/cuBLAS/blob/master/All_pairs_distances.cu
// Expects N x NDIMS matrix in points
thrust::device_vector<float> pairwise_dist(cublasHandle_t &handle, const thrust::device_vector<float> points, const unsigned int N, const unsigned int NDIMS) {
    const unsigned int BLOCKSIZE = 16;

    auto squared_vals = square(points, N * NDIMS);
    auto squared_norms = reduce_sum(handle, squared_vals, N, NDIMS, 1);
    
    float alpha = 1.f;
    float beta = 0.f;
    thrust::device_vector<float> dot_products(N * N);
    // Could replace this with cublasSsyrk, might be faster?
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, NDIMS, &alpha,
		                       thrust::raw_pointer_cast(points.data()), N, thrust::raw_pointer_cast(points.data()), N, &beta,
							   thrust::raw_pointer_cast(dot_products.data()), N));
 
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(iDivUp(N, BLOCKSIZE), iDivUp(N, BLOCKSIZE));
	assemble_final_result<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(squared_norms.data()), 
		                                         thrust::raw_pointer_cast(dot_products.data()), N);
    return dot_products;	
}

void gauss_normalize(cublasHandle_t &handle, thrust::device_vector<float> points, const unsigned int N, const unsigned int NDIMS) {
    auto means = reduce_mean(handle, points, N, NDIMS, 0);

    // zero center
    add_row_vec(points, means, N, NDIMS, -1.f);
    
    // compute standard deviation
    auto squared_vals = square(points, N * NDIMS);
    auto norm_sum_of_squares = reduce_alpha(handle, squared_vals, N, NDIMS, 1.f / (N - 1), 0);
    auto stddev = sqrt(norm_sum_of_squares, N * NDIMS);

    // normalize
    div_row_vec(points, stddev, N, NDIMS, 1.f);
}

thrust::device_vector<float> square(const thrust::device_vector<float> vec, const unsigned int N) {
    thrust::device_vector<float> squared_vals(N);
    thrust::transform(vec.begin(), vec.end(), squared_vals.begin(), func_square());
    return squared_vals;
}

thrust::device_vector<float> sqrt(const thrust::device_vector<float> vec, const unsigned int N) {
    thrust::device_vector<float> sqrt_vals(N);
    thrust::transform(vec.begin(), vec.end(), sqrt_vals.begin(), func_sqrt());
    return sqrt_vals;
}

// expects matrix of size N x M
thrust::device_vector<float> reduce_alpha(cublasHandle_t &handle, const thrust::device_vector<float> matrix, const unsigned int N, const unsigned int M, float alpha, const int axis) {
    if (axis == 0) {
        thrust::device_vector<float> ones(N, 1.f);
        thrust::device_vector<float> means(M);

        float beta = 0.f;
        cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, thrust::raw_pointer_cast(matrix.data()), N,
                                    thrust::raw_pointer_cast(ones.data()), 1, &beta, thrust::raw_pointer_cast(means.data()), 1));
        return means;
    } else if (axis == 1) {
        thrust::device_vector<float> ones(M, 1.f);
        thrust::device_vector<float> means(N);

        float beta = 0.f;
        cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, thrust::raw_pointer_cast(matrix.data()), N,
                                    thrust::raw_pointer_cast(ones.data()), 1, &beta, thrust::raw_pointer_cast(means.data()), 1));
        return means;
    } else {
        throw std::runtime_error("Axis must be 0 or 1.");
    }
}

thrust::device_vector<float> reduce_mean(cublasHandle_t &handle, const thrust::device_vector<float> matrix, const unsigned int N, const unsigned int M, const int axis) {
    float alpha = 1.f / N;
    return reduce_alpha(handle, matrix, N, M, alpha, axis);
}


thrust::device_vector<float> reduce_sum(cublasHandle_t &handle, const thrust::device_vector<float> matrix, const unsigned int N, const unsigned int M, const int axis) {
    float alpha = 1.f;
    return reduce_alpha(handle, matrix, N, M, alpha, axis);
}

int main(int argc, char **argv) {
    const unsigned int NDIMS = 50;
    const unsigned int N = 1 << 14;
    
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrices allocation and initialization
    thrust::device_vector<float> d_X(NDIMS * N);
    for (size_t i = 0; i < d_X.size(); i++) 
        d_X[i] = (float) dist(rng);

    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Starting pairwise distance calculation with %u points.\n", N);
    cudaEventRecord(start);
    auto distances = pairwise_dist(handle, d_X, N, NDIMS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f (ms)\n", milliseconds);
}

