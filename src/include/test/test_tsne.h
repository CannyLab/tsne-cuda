/**
* Tests for the TSNE functions
*/

void test_compute_pij(unsigned int N, unsigned int NDIMS) {

    std::default_random_engine generator;
    std::normal_distribution<double> distribution1(-10.0, 1.0);
    std::normal_distribution<double> distribution2(10.0, 1.0);

    std::vector<float> h_X(NDIMS * N);
    for (int i = 0; i < NDIMS * N; i ++) {
        if (i % N < (N / 2)) {
            h_X[i] = distribution1(generator);
        } else {
            h_X[i] = distribution2(generator);
        }
    }
    std::vector<float> sigmas(N);
    for (int i = 0; i < N; i++) {
        sigmas[i] = distribution1(generator);
    }
    compute_pij_cpu(h_X, sigmas, N, NDIMS);
    EXPECT_EQ(0, 0); 
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
