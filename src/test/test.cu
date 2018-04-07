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
#include <time.h>

#include "gtest/gtest.h"

void test_pairwise_distance(int N, int NDIM) {
    srand (time(NULL));

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
        //EXPECT_TRUE(abs(h_distances[i] - distances[i]) <= 1e-4);
        EXPECT_NEAR(h_distances[i],distances[i], 1e-4);
        EXPECT_TRUE(h_distances[i] >= 0);
    }
}

void test_tsne(unsigned int N,unsigned int NDIMS) {
    srand (time(NULL));

    std::default_random_engine generator;
    std::normal_distribution<double> distribution1(-10.0, 1.0);
    std::normal_distribution<double> distribution2(10.0, 1.0);

    thrust::host_vector<float> h_X(NDIMS * N);
    for (int i = 0; i < NDIMS * N; i ++) {
        if (i % N < (N / 2)) {
            h_X[i] = distribution1(generator);
        } else {
            h_X[i] = distribution2(generator);
        }
    }

    // --- Matrices allocation and initialization
    thrust::device_vector<float> d_X(NDIMS * N);
    thrust::copy(h_X.begin(), h_X.end(), d_X.begin());

    // for (int i = 0; i < NDIMS * N; i++) {
    //     std::cout << d_X[i] << " ";
    // }
    // printf("\n");
    
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

void test_reduce_sum_col(int N, int M) {

    // Construct the matrix with values in [0,1]
    std::mt19937 eng; 
    eng.seed(time(NULL));
    std::normal_distribution<float> dist;  
    float points[N * M];
    for (int i = 0; i < N * M; i++) points[i] = dist(eng);

    // Copy the matrix to the GPU
    thrust::device_vector<float> d_points(N*M);
    thrust::copy(points, points+(N*M), d_points.begin());

    // Construct CUBLAS handle
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = reduce_sum(handle, d_points, N, M, 0);

    // Copy the data back to the cpu
    float gpu_result[M];
    thrust::copy(d_sums.begin(), d_sums.end(), gpu_result);

    // Compute the reduction on the cpu
    float cpu_result[M];
    memset(cpu_result, 0, M*sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cpu_result[j] += points[i + j*N];
        }
    }

    for (int i = 0; i < M; i++) {
        EXPECT_NEAR(cpu_result[i], gpu_result[i], 1e-4);
    }
}

void test_reduce_sum_row(int N, int M) {

    // Construct the matrix with values in [0,1]
    std::mt19937 eng; 
    eng.seed(time(NULL));
    std::normal_distribution<float> dist;  
    float points[N * M];
    for (int i = 0; i < N * M; i++) points[i] = dist(eng);

    // Copy the matrix to the GPU
    thrust::device_vector<float> d_points(N*M);
    thrust::copy(points, points+(N*M), d_points.begin());

    // Construct CUBLAS handle
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = reduce_sum(handle, d_points, N, M, 1);

    // Copy the data back to the cpu
    float gpu_result[N];
    thrust::copy(d_sums.begin(), d_sums.end(), gpu_result);

    // Compute the reduction on the cpu
    float cpu_result[N];
    memset(cpu_result, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cpu_result[i] += points[i + j*N];
        }
    }

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(cpu_result[i], gpu_result[i], 1e-4);
    }
}

void test_reduce_mean_col(int N, int M) {

    // Construct the matrix with values in [0,1]
    std::mt19937 eng; 
    eng.seed(time(NULL));
    std::normal_distribution<float> dist;  
    float points[N * M];
    for (int i = 0; i < N * M; i++) points[i] = dist(eng);

    // Copy the matrix to the GPU
    thrust::device_vector<float> d_points(N*M);
    thrust::copy(points, points+(N*M), d_points.begin());

    // Construct CUBLAS handle
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = reduce_mean(handle, d_points, N, M, 0);

    // Copy the data back to the cpu
    float gpu_result[M];
    thrust::copy(d_sums.begin(), d_sums.end(), gpu_result);

    // Compute the reduction on the cpu
    float cpu_result[M];
    memset(cpu_result, 0, M*sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cpu_result[j] += points[i + j*N];
        }
    }

    for (int i = 0; i < M; i++) {
        EXPECT_NEAR(cpu_result[i]/((float) N), gpu_result[i], 1e-4);
    }
}

void test_reduce_mean_row(int N, int M) {

    // Construct the matrix with values in [0,1]
    std::mt19937 eng; 
    eng.seed(time(NULL));
    std::normal_distribution<float> dist;  
    float points[N * M];
    for (int i = 0; i < N * M; i++) points[i] = dist(eng);

    // Copy the matrix to the GPU
    thrust::device_vector<float> d_points(N*M);
    thrust::copy(points, points+(N*M), d_points.begin());

    // Construct CUBLAS handle
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = reduce_mean(handle, d_points, N, M, 1);

    // Copy the data back to the cpu
    float gpu_result[N];
    thrust::copy(d_sums.begin(), d_sums.end(), gpu_result);

    // Compute the reduction on the cpu
    float cpu_result[N];
    memset(cpu_result, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cpu_result[i] += points[i + j*N];
        }
    }

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(cpu_result[i]/((float) M), gpu_result[i], 1e-4);
    }
}

void test_reduce_alpha_col(int N, int M, float alpha) {

    // Construct the matrix with values in [0,1]
    std::mt19937 eng; 
    eng.seed(time(NULL));
    std::normal_distribution<float> dist;  
    float points[N * M];
    for (int i = 0; i < N * M; i++) points[i] = dist(eng);

    // Copy the matrix to the GPU
    thrust::device_vector<float> d_points(N*M);
    thrust::copy(points, points+(N*M), d_points.begin());

    // Construct CUBLAS handle
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = reduce_alpha(handle, d_points, N, M, alpha, 0);

    // Copy the data back to the cpu
    float gpu_result[M];
    thrust::copy(d_sums.begin(), d_sums.end(), gpu_result);

    // Compute the reduction on the cpu
    float cpu_result[M];
    memset(cpu_result, 0, M*sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cpu_result[j] += points[i + j*N];
        }
    }

    for (int i = 0; i < M; i++) {
        EXPECT_NEAR(cpu_result[i]*alpha, gpu_result[i], 1e-4);
    }
}

void test_reduce_alpha_row(int N, int M, float alpha) {

    // Construct the matrix with values in [0,1]
    std::mt19937 eng; 
    eng.seed(time(NULL));
    std::normal_distribution<float> dist;  
    float points[N * M];
    for (int i = 0; i < N * M; i++) points[i] = dist(eng);

    // Copy the matrix to the GPU
    thrust::device_vector<float> d_points(N*M);
    thrust::copy(points, points+(N*M), d_points.begin());

    // Construct CUBLAS handle
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = reduce_alpha(handle, d_points, N, M, alpha, 1);

    // Copy the data back to the cpu
    float gpu_result[N];
    thrust::copy(d_sums.begin(), d_sums.end(), gpu_result);

    // Compute the reduction on the cpu
    float cpu_result[N];
    memset(cpu_result, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cpu_result[i] += points[i + j*N];
        }
    }

    for (int i = 0; i < N; i++) {
        EXPECT_NEAR(cpu_result[i]*alpha, gpu_result[i], 1e-4);
    }
}

namespace {

    TEST(PairwiseDist, 4x2) {test_pairwise_distance(4,2);}
    TEST(PairwiseDist, 64x2) {test_pairwise_distance(64,2);}
    TEST(PairwiseDist, 64x64) {test_pairwise_distance(64,64);}
    TEST(PairwiseDist, 64x128) {test_pairwise_distance(64,128);}

    TEST(Reductions, ReduceSum_Col_512x512){test_reduce_sum_col(512,512);}
    TEST(Reductions, ReduceSum_Row_512x512){test_reduce_sum_row(512,512);}
    TEST(Reductions, ReduceMean_Col_512x512){test_reduce_mean_col(512,512);}
    TEST(Reductions, ReduceMean_Row_512x512){test_reduce_mean_row(512,512);}
    TEST(Reductions, ReduceAlpha_Col_512x512_pos){test_reduce_alpha_col(512,512,0.1);}
    TEST(Reductions, ReduceAlpha_Row_512x512_pos){test_reduce_alpha_row(512,512,0.1);}
    TEST(Reductions, ReduceAlpha_Col_512x512_neg){test_reduce_alpha_col(512,512,-0.1);}
    TEST(Reductions, ReduceAlpha_Row_512x512_neg){test_reduce_alpha_row(512,512, -0.1);}

    TEST(NaiveTSNE, 256x50) {test_tsne(256, 50);}
}
