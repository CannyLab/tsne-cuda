/**
 * @brief Implementation of locally sensitive hashing
 * 
 * @file lsh_utils.cu
 * @author David Chan
 * @date 2018-04-23
 */

#include "lsh_utils.h"

void compute_knns(unsigned int* knn_matrix, 
    float* points, 
    const unsigned int N_POINTS, 
    const unsigned int N_DIM, 
    const unsigned int K,
    const float R) 
{

    // We need to allocate memory on the GPU for the data, and the result
    float* d_points;
    unsigned int* d_knn_result;
    cudaMalloc((void**)&d_points, N_POINTS * N_DIM * sizeof(float));
    cudaMalloc((void**) &d_knn_result, N_POINTS * K * sizeof(unsigned int));

    // Copy the points to the GPU
    cudaMemcpy(d_points, points, N_POINTS*N_DIM*sizeof(float),cudaMemcpyHostToDevice);

    // Allocate the bounds for the W computation
    float* h_lower = new float[N_DIM];
    float* h_upper = new float[N_DIM];
    memset(h_lower, 0.0, N_DIM*sizeof(float));
    memset(h_upper, 1.0, N_DIM*sizeof(float));

    // Determine some bounds for the LSH (namely, the number of LSH tables)
    unsigned int LSH_L = 5;
    
    // Do the proximity computation
    cudaDeviceSynchronize();
    compute_proximity_lsh(d_knn_result, d_points, N_POINTS, N_DIM, K, LSH_L, h_upper, h_lower);
    cudaDeviceSynchronize();

    // Copy the gpu knn results back
    cudaMemcpy(knn_matrix, d_knn_result, N_POINTS*K*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_points);
    cudaFree(d_knn_result);
}

void compute_proximity_lsh(unsigned int *d_knn_result, 
                            float* d_points, 
                            const unsigned int N_POINTS, 
                            const unsigned int N_DIM, 
                            const unsigned int K, 
                            const unsigned int L,
                            float* h_upper,
                            float* h_lower)
{

    // Setup some constant parameters
    const unsigned int LSH_nP = 32;

    // Construct some constant parameters which will be used to compute the gaussians
    int* h_P[LSH_nP];
    for (int i = 0; i < LSH_nP; i++) h_P[i] = rand();
    // Move these parameters to the GPU
    __device__ __constant__ d_P[LSH_nP];
    cudaMemcpyToSymbol(d_P, h_P, LSH_nP * sizeof(int), 0, cudaMemcpyHostToDevice);

    // Compute the W value
    float* limits[N_DIM];
    for (unsigned int i = 0; i < N_DIM; i++) {
        if (fabs(h_upper[i]) > fabs(h_lower[i])) limits[i] = fabs(h_upper[i]);
        else limits[i] = fabs(h_lower[i]);
    }
    float W = 0;
    for (unsigned int i = 0; i < N_DIM; i++) W += limits[i];
    W = W * sqrt(float(N_DIM)) / 4.0;
    W /= powf(N_POINTS / 10000.0f, 1.0f / N_DIM);

    // Allocate memory for the LSH 
    float* 


    // Actually compute the LSH values

    
}

