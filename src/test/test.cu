/**
 * @brief Unit Tests for the CUDA-TSNE Library
 * 
 * @file test.cu
 * @author David Chan
 * @date 2018-04-04
 */

#include "common.h"
#include "util/cuda_utils.h"
#include "util/random_utils.h"
#include "util/distance_utils.h"
#include "naive_tsne.h"

#include "gtest/gtest.h"

namespace {

    TEST(PairwiseDist, 4x2) {
        const int N = 4;
        const int NDIM = 2;
        // Create some random points in 2 dimensions
        float points[N * NDIM];
        for (int i = 0; i < N*NDIM; i++) {
            points[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        }

        // Construct the thrust vector
        thrust::host_vector<float> h_points(N*NDIM);
        for (int i = 0; i < N*NDIM; i++) 
            h_points[i] = points[i]; 

        thrust::device_vector<float> d_points(N*NDIM);
        thrust::copy(h_points.begin(), h_points.end(), d_points.begin());
        thrust::device_vector<float> d_distances(N*N);

        // Construct the CUBLAS handle
        cublasHandle_t handle;
        cublasSafeCall(cublasCreate(&handle));
        pairwise_dist(handle, d_distances, d_points, N, NDIM);

        thrust::host_vector<float> h_distances(N*N);
        thrust::copy(d_distances.begin(), d_distances.end(), h_distances.begin());

        // Compute the pairwise distances on our own
        float distances[N*N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < NDIM; k++) {
                    sum += (points[i + k*N] - points[j + k*N]) * (points[i + k*N] - points[j + k*N]);
                }
                distances[i*N + j] = sqrt(sum);
            }
        }

        // Compare
        for (int i = 0; i < N*N; i++){
            EXPECT_EQ((int) (h_distances[i]*1e5), (int) (distances[i]*1e5));
            EXPECT_TRUE(h_distances[i] >= 0);
        }
            
    }

    TEST(NaiveTSNE, 256) {
        const unsigned int NDIMS = 50;
        const unsigned int N = 1 << 5;
        
        thrust::default_random_engine rng;
        thrust::uniform_int_distribution<int> dist(10, 99);
    
        // --- Matrices allocation and initialization
        thrust::device_vector<float> d_X(NDIMS * N);
        for (size_t i = 0; i < d_X.size(); i++) 
            d_X[i] = (float) dist(rng);
    
        thrust::device_vector<float> sigmas(N, 1.0f);
        cublasHandle_t handle;
        cublasSafeCall(cublasCreate(&handle));
    
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        printf("Starting TSNE calculation with %u points.\n", N);
        cudaEventRecord(start);
        naive_tsne(handle, d_X, N, NDIMS);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Elapsed time: %f (ms)\n", milliseconds);
        EXPECT_EQ(0, 0);
    }
    // TEST(NaiveTSNE, 512) {
    //     const unsigned int NDIMS = 50;
    //     const unsigned int N = 1 << 9;
        
    //     thrust::default_random_engine rng;
    //     thrust::uniform_int_distribution<int> dist(10, 99);
    
    //     // --- Matrices allocation and initialization
    //     thrust::device_vector<float> d_X(NDIMS * N);
    //     for (size_t i = 0; i < d_X.size(); i++) 
    //         d_X[i] = (float) dist(rng);
    
    //     thrust::device_vector<float> sigmas(N, 1.0f);
    //     cublasHandle_t handle;
    //     cublasSafeCall(cublasCreate(&handle));
    
    //     cudaEvent_t start, stop;
    //     cudaEventCreate(&start);
    //     cudaEventCreate(&stop);
    //     printf("Starting TSNE calculation with %u points.\n", N);
    //     cudaEventRecord(start);
    //     naive_tsne(handle, d_X, N, NDIMS);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     float milliseconds = 0;
    //     cudaEventElapsedTime(&milliseconds, start, stop);
    //     printf("Elapsed time: %f (ms)\n", milliseconds);
    //     EXPECT_EQ(0, 0);
    // }
    // TEST(NaiveTSNE, 16k) {
    //     const unsigned int NDIMS = 50;
    //     const unsigned int N = 1 << 14;
        
    //     thrust::default_random_engine rng;
    //     thrust::uniform_int_distribution<int> dist(10, 99);
    
    //     // --- Matrices allocation and initialization
    //     thrust::device_vector<float> d_X(NDIMS * N);
    //     for (size_t i = 0; i < d_X.size(); i++) 
    //         d_X[i] = (float) dist(rng);
    
    //     thrust::device_vector<float> sigmas(N, 1.0f);
    //     cublasHandle_t handle;
    //     cublasSafeCall(cublasCreate(&handle));
    
    //     cudaEvent_t start, stop;
    //     cudaEventCreate(&start);
    //     cudaEventCreate(&stop);
    //     printf("Starting TSNE calculation with %u points.\n", N);
    //     cudaEventRecord(start);
    //     naive_tsne(handle, d_X, N, NDIMS);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     float milliseconds = 0;
    //     cudaEventElapsedTime(&milliseconds, start, stop);
    //     printf("Elapsed time: %f (ms)\n", milliseconds);
    //     EXPECT_EQ(0, 0);
    // }
}
