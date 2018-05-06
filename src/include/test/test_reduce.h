/**
 * @brief Unit Tests for the reduce functions
 * 
 * @file test_reduce.h
 * @author David Chan
 * @date 2018-04-11
 */

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
    CublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = tsnecuda::util::ReduceSum(handle, d_points, N, M, 0);

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
    CublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = tsnecuda::util::ReduceSum(handle, d_points, N, M, 1);

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
    CublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = tsnecuda::util::ReduceMean(handle, d_points, N, M, 0);

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
    CublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = tsnecuda::util::ReduceMean(handle, d_points, N, M, 1);

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
    CublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = tsnecuda::util::ReduceAlpha(handle, d_points, N, M, alpha, 0);

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
    CublasSafeCall(cublasCreate(&handle));

    // Do the reduction
    auto d_sums = tsnecuda::util::ReduceAlpha(handle, d_points, N, M, alpha, 1);

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