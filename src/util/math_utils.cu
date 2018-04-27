/**
 * @brief 
 * 
 * @file math_utils.cu
 * @author David Chan
 * @date 2018-04-04
 */

 #include "util/math_utils.h"

 struct func_square {
    __host__ __device__ float operator()(const float &x) const { return x * x; }
};
struct func_sqrt {
    __host__ __device__ float operator()(const float &x) const { return pow(x, 0.5); }
};
struct func_abs {
    __host__ __device__ float operator()(const float &x) const { return fabs(x); }
};
struct func_nan_or_inf {
    __host__ __device__ bool operator()(const float &x) const { return isnan(x) || isinf(x); }
};
struct func_cast_to_int {
    __host__ __device__ int operator()(const long &x) const { return (int) x; }
};

void Math::gauss_normalize(cublasHandle_t &handle, thrust::device_vector<float> &points, const unsigned int N, const unsigned int NDIMS) {
    auto means = Reduce::reduce_mean(handle, points, N, NDIMS, 0);

    // zero center
    Broadcast::broadcast_matrix_vector(points, means, N, NDIMS, thrust::minus<float>(), 1, 1.f);
    
    // compute standard deviation
    thrust::device_vector<float> squared_vals(points.size());
    Math::square(points, squared_vals);
    auto norm_sum_of_squares = Reduce::reduce_alpha(handle, squared_vals, N, NDIMS, 1.f / (N - 1), 0);
    thrust::device_vector<float> stddev(norm_sum_of_squares.size());
    Math::sqrt(norm_sum_of_squares, stddev);

    // normalize
    Broadcast::broadcast_matrix_vector(points, stddev, N, NDIMS, thrust::divides<float>(), 1, 1.f);
}

void Math::square(const thrust::device_vector<float> &vec, thrust::device_vector<float> &out) {
    thrust::transform(vec.begin(), vec.end(), out.begin(), func_square());
}

void Math::sqrt(const thrust::device_vector<float> &vec, thrust::device_vector<float> &out) {
    thrust::transform(vec.begin(), vec.end(), out.begin(), func_sqrt());
}

float Math::norm(const thrust::device_vector<float> &vec) {
    return std::sqrt( thrust::transform_reduce(vec.begin(), vec.end(), func_square(), 0.0f, thrust::plus<float>()) );
}

bool Math::any_nan_or_inf(const thrust::device_vector<float> &vec) {
    return thrust::transform_reduce(vec.begin(), vec.end(), func_nan_or_inf(), 0, thrust::plus<bool>());
}

void Math::max_norm(thrust::device_vector<float> &vec) {
    float max_val = thrust::transform_reduce(vec.begin(), vec.end(), func_abs(), 0.0f, thrust::maximum<float>());
    thrust::constant_iterator<float> div_iter(max_val);
    thrust::transform(vec.begin(), vec.end(), div_iter, vec.begin(), thrust::divides<float>());
}

void Sparse::sym_mat_gpu(float* values, int* indices, thrust::device_vector<float> &sym_values,  
                                thrust::device_vector<int> &sym_colind, thrust::device_vector<int> &sym_rowptr, int* sym_nnz, 
                                unsigned int N_POINTS, unsigned int K) {
    // Allocate memory
    // std::cout << "Allocating initial memory on GPU..." << std::endl;
    int* csrRowPtrA = nullptr;
    cudaMalloc((void**)& csrRowPtrA, (N_POINTS+1)*sizeof(int));
    int* csrColPtrA = nullptr;
    cudaMalloc((void**)& csrColPtrA, (N_POINTS*K)*sizeof(int));
    float* csrValA = nullptr;
    cudaMalloc((void**)& csrValA, (N_POINTS*K)*sizeof(float));

    // Copy the data
    cudaMemcpy(csrColPtrA, indices, N_POINTS*K*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrValA, values, N_POINTS*K*sizeof(float), cudaMemcpyHostToDevice);


    thrust::device_vector<int> vx(csrRowPtrA, csrRowPtrA+N_POINTS+1);
    thrust::sequence(vx.begin(), vx.end(), 0,(int) K);
    thrust::copy(vx.begin(), vx.end(), csrRowPtrA);
    cudaDeviceSynchronize();
    *sym_nnz = -1; // Initialize the default value of the sym_nnz var

    // Initialize the cusparse handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Initialize the matrix descriptor
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO); 

    // Sort the matrix properly
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;
    int *P = NULL;

    // step 1: allocate buffer
    cusparseXcsrsort_bufferSizeExt(handle, N_POINTS, N_POINTS, N_POINTS*K, csrRowPtrA, csrColPtrA, &pBufferSizeInBytes);
    cudaDeviceSynchronize();
    cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes);

    // step 2: setup permutation vector P to identity
    cudaMalloc( (void**)&P, sizeof(int)*N_POINTS*K);
    cusparseCreateIdentityPermutation(handle, N_POINTS*K, P);
    cudaDeviceSynchronize();

    // step 3: sort CSR format
    cusparseXcsrsort(handle, N_POINTS, N_POINTS, N_POINTS*K, descr, csrRowPtrA, csrColPtrA, P, pBuffer);
    cudaDeviceSynchronize();

    // step 4: gather sorted csrVal
    float* csrValA_sorted = nullptr;
    cudaMalloc((void**)& csrValA_sorted, (N_POINTS*K)*sizeof(float));
    cusparseSgthr(handle, N_POINTS*K, csrValA, csrValA_sorted, P, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    // Free some memory
    cudaFree(pBuffer);
    cudaFree(P);
    cudaFree(csrValA);
    csrValA = csrValA_sorted;

    // We need A^T, so we do a csr2csc() call
    // std::cout << "Allocating memory for transpose..." << std::endl;
    int* cscRowIndAT = nullptr;
    cudaMalloc((void**)& cscRowIndAT, (N_POINTS*K)*sizeof(int));
    int* cscColPtrAT = nullptr;
    cudaMalloc((void**)& cscColPtrAT, (N_POINTS+1)*sizeof(int));
    float* cscValAT = nullptr;
    cudaMalloc((void**)& cscValAT, (N_POINTS*K)*sizeof(float));

    // Do the transpose operation
    cusparseScsr2csc(handle, N_POINTS , N_POINTS , K*N_POINTS , csrValA, csrRowPtrA, csrColPtrA, cscValAT, 
                cscRowIndAT, cscColPtrAT, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    // Now compute the output size of the matrix
    int baseC, nnzC;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    sym_rowptr.resize(N_POINTS+1);
    // cudaMalloc((void**) sym_rowptr, sizeof(int)*(N_POINTS+1));
    cusparseXcsrgeamNnz(handle, N_POINTS, N_POINTS,
                            descr, N_POINTS*K, csrRowPtrA, csrColPtrA,
                            descr, N_POINTS*K, cscColPtrAT, cscRowIndAT,
                            descr, thrust::raw_pointer_cast(sym_rowptr.data()), sym_nnz
                        );
    cudaDeviceSynchronize();

    if (-1 != *sym_nnz) {
        nnzC = *sym_nnz;
    } else {
        cudaMemcpy(&nnzC, thrust::raw_pointer_cast(sym_rowptr.data())+N_POINTS,sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, thrust::raw_pointer_cast(sym_rowptr.data()), sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Allocate memory for the new summed array
    sym_colind.resize(nnzC);
    sym_values.resize(nnzC);
    // cudaMalloc((void**) sym_colind, sizeof(int)*nnzC);
    // cudaMalloc((void**) sym_values, sizeof(float)*nnzC);

    // Sum the arrays
    // std::cout << "Symmetrizing..." << std::endl;
    float alpha = 0.5f;
    float beta = 0.5f;
    cusparseScsrgeam(handle, N_POINTS, N_POINTS, 
       &alpha, descr, N_POINTS*K, csrValA, csrRowPtrA, csrColPtrA,
        &beta, descr, N_POINTS*K, cscValAT, cscColPtrAT, cscRowIndAT,
        descr, thrust::raw_pointer_cast(sym_values.data()), 
                thrust::raw_pointer_cast(sym_rowptr.data()), 
                thrust::raw_pointer_cast(sym_colind.data())
    );
    cudaDeviceSynchronize();

    // Free the memory we were using...
    cudaFree(csrValA);
    cudaFree(cscValAT);
    cudaFree(csrRowPtrA);
    cudaFree(cscColPtrAT);
    cudaFree(csrColPtrA);
    cudaFree(cscRowIndAT);
}

