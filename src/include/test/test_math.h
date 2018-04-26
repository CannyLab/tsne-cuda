/**
 * @brief Unit Tests for the math functions
 * 
 * @file test_math.h
 * @author David Chan
 * @date 2018-04-25
 */


void test_sym_mat(int N, int K) {
    // Construct a random matrix
    float* mat_val = new float[N*K];
    int* indices = new int[N*K];

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist(0, 1.0);


    for (int i = 0; i < N; i++) {
        
        // Get a random subset of rows to fill
        std::vector<unsigned int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::random_shuffle(idx.begin(), idx.end());

        for (int k = 0; k < K; k++) {
            mat_val[i*K + k] = dist(e2);
            indices[i*K + k] = idx[k];
        }
    }

     // Symmetrize the matrix on the CPU
    float* mat_val_sim = new float[N*N];
    memset(mat_val_sim, 0.0, N*N*sizeof(float));
    
    // Densify the matrix
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            mat_val_sim[i*N + indices[i*K + k]] += mat_val[i*K + k];
        }
    }
    // Symmetrize the matrix
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            mat_val_sim[i*N + j] = 0.5*(mat_val_sim[i*N + j] + mat_val_sim[j*N+i]);
            mat_val_sim[j*N + i] = mat_val_sim[i*N + j];
        }
    }

    // Symmetrize the matrix on the GPU
    float* gpu_sym = nullptr;
    int* gpu_col_idx = nullptr;
    int* gpu_row_idx = nullptr;
    int sym_nnz = -1;

    Sparse::sym_mat_gpu(mat_val, indices, &gpu_sym, &gpu_col_idx, &gpu_row_idx, &sym_nnz, N, K);


    // Copy the data back
    float* host_data = new float[sym_nnz];
    int* host_row_idx = new int[N+1];
    int* host_col_idx = new int[sym_nnz];

    cudaMemcpy(host_data, gpu_sym, sym_nnz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_row_idx, gpu_row_idx, (N+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_col_idx, gpu_col_idx, sym_nnz*sizeof(int), cudaMemcpyDeviceToHost);

    // Densify the host matrix
    float mat[N*N];
    memset(mat, 0, N*N*sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = host_row_idx[i]; j < host_row_idx[i+1];j++) {
            mat[i*N + host_col_idx[j]] = host_data[j];
        }
    }

    // Assert that the two dense representations are equal
    for (int i = 0; i < N*N; i++) ASSERT_NEAR(mat[i], mat_val_sim[i], 1e-4);

    cudaFree(gpu_sym);
    cudaFree(gpu_col_idx);
    cudaFree(gpu_row_idx);

    delete mat_val;
    delete indices;
    delete mat_val_sim;

}