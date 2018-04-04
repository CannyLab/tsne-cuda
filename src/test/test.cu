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
#include "naive_tsne.h"

#include "gtest/gtest.h"

namespace {
    TEST(SampleTest, Sample1) {
        EXPECT_EQ(17, 17);
    }

    TEST(NaiveTSNE, NPoints256) {
        const unsigned int NDIMS = 50;
        const unsigned int N = 1 << 8;
        
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
}