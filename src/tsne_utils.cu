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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include "tsne_utils.cuh"
#include "Utilities.cuh"

struct abs2 {
	__host__ __device__ double operator()(const float &x) const { return x * x; }
};

__global__ void assemble_final_result(const float * __restrict__ d_norms_x_2, float * __restrict__ d_dots,
									  const int N) {

	const int i = threadIdx.x + blockIdx.x * gridDim.x;
	const int j = threadIdx.y + blockIdx.y * gridDim.y;

	if ((i < N) && (j < N)) d_dots[i * N + j] = d_norms_x_2[j] + d_norms_x_2[i] - 2 * d_dots[i * N + j];
    
}

// Code from https://github.com/OrangeOwlSolutions/cuBLAS/blob/master/All_pairs_distances.cu
thrust::device_vector<float> pairwise_dist(const thrust::device_vector<float> data, const unsigned int N, const unsigned int NDIMS) {
    const unsigned int BLOCKSIZE = 16;

    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    thrust::device_vector<float> squared_norms(N);
    thrust::device_vector<float> squared_vals(NDIMS * N);
    thrust::transform(data.begin(), data.end(), squared_vals.begin(), abs2());

    thrust::device_vector<float> ones(NDIMS, 1.f);

    float alpha = 1.f;
    float beta = 0.f;
    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, NDIMS, N, &alpha, thrust::raw_pointer_cast(squared_vals.data()), NDIMS,
                                thrust::raw_pointer_cast(ones.data()), 1, &beta, thrust::raw_pointer_cast(squared_norms.data()), 1));

    thrust::device_vector<float> dot_products(N * N);
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, NDIMS, &alpha,
		                       thrust::raw_pointer_cast(data.data()), NDIMS, thrust::raw_pointer_cast(data.data()), NDIMS, &beta,
							   thrust::raw_pointer_cast(dot_products.data()), N));
 
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(iDivUp(N, BLOCKSIZE), iDivUp(N, BLOCKSIZE));
	assemble_final_result<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(squared_norms.data()), 
		                                         thrust::raw_pointer_cast(dot_products.data()), N);
    return dot_products;
	
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Starting pairwise distance calculation with %u points.\n", N);
    cudaEventRecord(start);
    auto distances = pairwise_dist(d_X, N, NDIMS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f (ms)\n", milliseconds);
}

