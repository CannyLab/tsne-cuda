/**
 * @brief Unit Tests for the T-SNE functions
 * 
 * @file test_tsne.h
 * @author David Chan
 * @date 2018-04-11
 */

void test_cpu_compute_pij(unsigned int N, unsigned int NDIMS) {

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
    std::vector<float> pij = compute_pij_cpu(h_X, sigmas, N, NDIMS);
    float first_prob = 0.0f;
    for (int i = 1; i < N; i++) {
        first_prob += pij[i];
    }
    ASSERT_EQ((int) (first_prob*1000), 1000); 
}


void test_cpu_sigmas_search(unsigned int N, unsigned int NDIMS) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution1(-5.0, 1.0);
    std::normal_distribution<double> distribution2(5.0, 1.0);
    std::vector<float> h_X(NDIMS * N);
    for (int i = 0; i < NDIMS * N; i ++) {
        if (i % N < (N / 2)) {
            h_X[i] = distribution1(generator);
        } else {
            h_X[i] = distribution2(generator);
        }
    }
    std::vector<float> sigmas = sigmas_search_cpu(h_X, N, NDIMS, 8.0);
    // for (int i = 0; i < N; i++) {
    //     std::cout << sigmas[i] << " ";
    // }
    // printf("\n");
    ASSERT_EQ(0, 0);
}


void test_cpu_is_gpu_perplexity(unsigned int N, unsigned int NDIMS) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution1(-5.0, 1.0);
    std::normal_distribution<double> distribution2(5.0, 1.0);
    std::vector<float> h_X(NDIMS * N);
    for (int i = 0; i < NDIMS * N; i ++) {
        if (i % N < (N / 2)) {
            h_X[i] = distribution1(generator);
        } else {
            h_X[i] = distribution2(generator);
        }
    }
    std::vector<float> sigmas(N);
    std::uniform_real_distribution<double> udist(10.0, 20.0);
    for (int i = 0; i < N; i++) {
        // sigmas[i] = udist(generator);
        sigmas[i] = 10.0;
    }

    // Copy points to GPU
    thrust::device_vector<float> d_X(NDIMS * N);
    thrust::device_vector<float> d_sigmas(N);
    thrust::device_vector<float> d_perp(N);
    thrust::device_vector<float> lbs(N);
    thrust::device_vector<float> ubs(N);
    thrust::copy(h_X.begin(), h_X.end(), d_X.begin());
    thrust::copy(sigmas.begin(), sigmas.end(), d_sigmas.begin());

    // Compute the CPU Perplexity
    std::vector<float> cpu_pij = compute_pij_cpu(h_X, sigmas, N, NDIMS);
    float h_perp = get_perplexity(cpu_pij, 0, N);

    // Compute the GPU Pij
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));
    thrust::device_vector<float> d_gpu_pij(N*N);
    NaiveTSNE::compute_pij(handle, d_gpu_pij, d_X, d_sigmas, N, NDIMS);
    NaiveTSNE::thrust_search_perplexity(handle, d_sigmas, lbs, ubs, d_perp, d_gpu_pij, 8.0, N);

    cudaDeviceSynchronize();
    float gpu_perp[N];
    thrust::copy(d_perp.begin(), d_perp.end(), gpu_perp);
    // std::cout << h_perp << ", " << gpu_perp[0] << "\n";

    ASSERT_EQ((int) (gpu_perp[0]*1e4), (int) (h_perp*1e4));
}
void test_cpu_is_gpu_pij(unsigned int N, unsigned int NDIMS) {

    // Initialize the points
    std::default_random_engine generator;
    std::normal_distribution<double> distribution1(-5.0, 1.0);
    std::normal_distribution<double> distribution2(5.0, 1.0);
    std::vector<float> h_X(NDIMS * N);
    for (int i = 0; i < NDIMS * N; i ++) {
        if (i % N < (N / 2)) {
            h_X[i] = distribution1(generator);
        } else {
            h_X[i] = distribution2(generator);
        }
    }
    std::vector<float> sigmas(N);
    std::uniform_real_distribution<double> udist(10.0, 20.0);
    for (int i = 0; i < N; i++) {
        // sigmas[i] = udist(generator);
        sigmas[i] = 1.0;
    }

    // Copy points to GPU
    thrust::device_vector<float> d_X(NDIMS * N);
    thrust::device_vector<float> d_sigmas(N);
    thrust::copy(h_X.begin(), h_X.end(), d_X.begin());
    thrust::copy(sigmas.begin(), sigmas.end(), d_sigmas.begin());

    // Compute the CPU Pij
    std::vector<float> cpu_pij = compute_pij_cpu(h_X, sigmas, N, NDIMS);

    // Compute the GPU Pij
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));
    thrust::device_vector<float> d_gpu_pij(N*N);
    NaiveTSNE::compute_pij(handle, d_gpu_pij, d_X, d_sigmas, N, NDIMS);
    cudaDeviceSynchronize();
    float gpu_pij[N*N];
    thrust::copy(d_gpu_pij.begin(), d_gpu_pij.end(), gpu_pij);
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < NDIMS; j++) {
    //          std::cout << h_X[i + j * N] << ", ";
    //     }
    //     printf("\n");
    // }
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //          std::cout << cpu_pij[i * N + j] << " ";
    //     }
    //     printf("\n");
    // }
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //          std::cout << gpu_pij[i * N + j] << " ";
    //     }
    //     printf("\n");
    // }

    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++){
            ASSERT_EQ((int) (cpu_pij[i + j*N]*1e4), (int) (gpu_pij[i*N + j]*1e4) );
        }
         
}

void test_sigmas_search(unsigned int N, unsigned int NDIMS) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution1(-5.0, 1.0);
    std::normal_distribution<double> distribution2(5.0, 1.0);
    std::vector<float> h_X(NDIMS * N);
    for (int i = 0; i < NDIMS * N; i ++) {
        if (i % N < (N / 2)) {
            h_X[i] = distribution1(generator);
        } else {
            h_X[i] = distribution2(generator);
        }
    }
    // Copy points to GPU
    thrust::device_vector<float> d_X(NDIMS * N);
    thrust::copy(h_X.begin(), h_X.end(), d_X.begin());
    float perplexity_target = 8.0;
    float eps = 1e-2;

    // Compute the GPU Pij
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));
    auto d_sigmas = NaiveTSNE::search_perplexity(handle, d_X, perplexity_target, eps, N, NDIMS);
    std::vector<float> sigmas = sigmas_search_cpu(h_X, N, NDIMS, perplexity_target);
    // for (int i = 0; i < N; i++) {
    //    std::cout << sigmas[i] << " ";
    // }
    // printf("\n");
    // for (int i = 0; i < N; i++) {
    //    std::cout << d_sigmas[i] << " ";
    // }
    // printf("\n");
    
    cudaDeviceSynchronize();
    ASSERT_EQ(0, 0);    
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

    thrust::device_vector<float> sigmas(N, 1.0f);
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Starting TSNE calculation with %u points.\n", N);
    cudaEventRecord(start);
    NaiveTSNE::tsne(handle, d_X, N, NDIMS, 2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f (ms)\n", milliseconds);
    EXPECT_EQ(0, 0);
}


void test_bhtsne(int N, int NDIMS) {
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
    cublasHandle_t dense_handle;
    cublasSafeCall(cublasCreate(&dense_handle));
    cusparseHandle_t sparse_handle;
    cusparseSafeCall(cusparseCreate(&sparse_handle));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Starting TSNE calculation with %u points.\n", N);
    cudaEventRecord(start);
    BHTSNE::tsne(dense_handle, sparse_handle, thrust::raw_pointer_cast(h_X.data()), N, NDIMS, 2, 2, 1.0, 0.0, 1000, 1000, 0.0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f (ms)\n", milliseconds);
    EXPECT_EQ(0, 0);
}
