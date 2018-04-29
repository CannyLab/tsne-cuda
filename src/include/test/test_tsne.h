/**
 * @brief Unit Tests for the T-SNE functions
 * 
 * @file test_tsne.h
 * @author David Chan
 * @date 2018-04-11
 */

#include <stdint.h>
#include <fstream>


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
        if (i < ((N / 2) * NDIMS)) {
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
    BHTSNE::tsne(dense_handle, sparse_handle, thrust::raw_pointer_cast(h_X.data()), N, NDIMS, 2, 45.0, 100.0, 2.0, 1000, 1000, 0.0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f (ms)\n", milliseconds);
    EXPECT_EQ(0, 0);
}


void test_bhtsne_mnist(std::string fname) {
    srand (time(NULL));

    std::default_random_engine generator;
    std::normal_distribution<double> distribution1(-10.0, 1.0);
    std::normal_distribution<double> distribution2(10.0, 1.0);
    int N = 2500;
    int NDIMS = 768;
    thrust::host_vector<float> h_X = Data::load_file(fname);

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

void test_bhtsne_ref(int N, int NDIMS) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution1(0.0, 20.0);
    std::normal_distribution<double> distribution2(0.0, 20.0);

    std::vector<float> h_X(NDIMS * N);
    for (int i = 0; i < N * NDIMS; i ++) {
        if (i < (N * NDIMS / 2)) {
            h_X[i] = distribution1(generator);
        } else {
            h_X[i] = distribution2(generator);
        }
    }
    float * fYs = (float *) calloc(N * 2, sizeof(float));

    for (int i = 0; i < N * 2; i++) {
        fYs[i] = (float) distribution1(generator);
    }

    int K = N - 1;

    double * forces = BHTSNERef::computeEdgeForces(h_X.data(), fYs, NDIMS, 2, N, K, 1.0f);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < NDIMS; j++)
            std::cout << forces[i * NDIMS + j] << " ";
        printf("\n");
    }


}

void test_bhtsne_full_mnist(std::string fname) {
    srand (time(NULL));
    // Construct the file stream
    std::ifstream mnist_data_file(fname, std::ios::in | std::ios::binary);
    
    // Read the data header
    std::cout << "Reading file header..." << std::endl;
    int magic_number = 0;
    mnist_data_file.read(reinterpret_cast<char *>(&magic_number), sizeof(int));
    if (mnist_data_file.gcount() != 4) {std::cout << "File read error (magic number). The number was: " << magic_number << std::endl;ASSERT_EQ(1, 0);}
    magic_number = ((magic_number>>24)&0xff) | // move byte 3 to byte 0
                    ((magic_number<<8)&0xff0000) | // move byte 1 to byte 2
                    ((magic_number>>8)&0xff00) | // move byte 2 to byte 1
                    ((magic_number<<24)&0xff000000); // byte 0 to byte 3
    if (magic_number != 2051) {std::cout << "Invalid magic number. The number was: " << magic_number << std::endl;ASSERT_EQ(1, 0);}
    
    int num_images = 0;
    mnist_data_file.read(reinterpret_cast<char *>(&num_images), sizeof(int));
    if (mnist_data_file.gcount() != 4) {std::cout << "File read error (number of images)." << std::endl;ASSERT_EQ(1, 0);}

    num_images = ((num_images>>24)&0xff) | // move byte 3 to byte 0
                    ((num_images<<8)&0xff0000) | // move byte 1 to byte 2
                    ((num_images>>8)&0xff00) | // move byte 2 to byte 1
                    ((num_images<<24)&0xff000000); // byte 0 to byte 3
    std::cout << "Num Images: " << num_images << std::endl;

    int num_rows = 0;
    mnist_data_file.read(reinterpret_cast<char *>(&num_rows), sizeof(int));
    if (mnist_data_file.gcount() != 4) {std::cout << "File read error (number of rows)." << std::endl;ASSERT_EQ(1, 0);}

    num_rows = ((num_rows>>24)&0xff) | // move byte 3 to byte 0
                    ((num_rows<<8)&0xff0000) | // move byte 1 to byte 2
                    ((num_rows>>8)&0xff00) | // move byte 2 to byte 1
                    ((num_rows<<24)&0xff000000); // byte 0 to byte 3
    std::cout << "Num Rows: " << num_rows << std::endl;

    int num_columns = 0;
    mnist_data_file.read(reinterpret_cast<char *>(&num_columns), sizeof(int));
    if (mnist_data_file.gcount() != 4) {std::cout << "File read error (number of columns)." << std::endl;ASSERT_EQ(1, 0);}

    num_columns = ((num_columns>>24)&0xff) | // move byte 3 to byte 0
                    ((num_columns<<8)&0xff0000) | // move byte 1 to byte 2
                    ((num_columns>>8)&0xff00) | // move byte 2 to byte 1
                    ((num_columns<<24)&0xff000000); // byte 0 to byte 3
    std::cout << "Num Cols: " << num_columns << std::endl;

    std::cout << "Reading pixels from file..." << std::endl;
    uint8_t pixel_val = 0;
    float* data = new float[num_images*num_rows*num_columns];
    for (int idx = 0; idx < num_images; idx++) {
        for (int jdx = 0; jdx < num_rows*num_columns; jdx++) {
            mnist_data_file.read(reinterpret_cast<char *>(&pixel_val), sizeof(uint8_t));
            if (mnist_data_file.gcount() != 1) {std::cout << "File read error (pixel)." << std::endl;ASSERT_EQ(1, 0);}
            data[idx*num_rows*num_columns + jdx] = ((float) pixel_val) / 255.0f;
        }
    }

    std::cout << "Done reading!" << std::endl;
    mnist_data_file.close();

    // --- Matrices allocation and initialization
    cublasHandle_t dense_handle;
    cublasSafeCall(cublasCreate(&dense_handle));
    cusparseHandle_t sparse_handle;
    cusparseSafeCall(cusparseCreate(&sparse_handle));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Starting TSNE calculation with %u points.\n", num_images);
    cudaEventRecord(start);
    BHTSNE::tsne(dense_handle, sparse_handle, data, num_images, num_columns*num_rows, 2, 45.0, 200.0, 12.0, 1000, 1000, 0.0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f (ms)\n", milliseconds);
    EXPECT_EQ(0, 0);

    delete[] data;


}

