/**
 * @brief Implementation of the math_utils.h file
 * 
 * @file math_utils.cu
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#include "util/math_utils.h"

void tsnecuda::util::GaussianNormalizeDeviceVector(cublasHandle_t &handle,
        thrust::device_vector<float> &d_points, const int num_points,
        const int num_dims) {
    // Compute the means
    auto d_means = tsnecuda::util::ReduceMean(handle, d_points, num_points,
                                         num_dims, 0);

    // Zero-Center
    tsnecuda::util::BroadcastMatrixVector(d_points, d_means, num_points, num_dims,
                                       thrust::minus<float>(), 1, 1.f);

    // Compute the standard deviation
    thrust::device_vector<float> squared_vals(d_points.size());
    tsnecuda::util::SquareDeviceVector(squared_vals, d_points);
    auto norm_sum_of_squares = tsnecuda::util::ReduceAlpha(handle, squared_vals,
            num_points, num_dims, 1.f / (num_points - 1), 0);
    thrust::device_vector<float> standard_deviation(norm_sum_of_squares.size());
    tsnecuda::util::SqrtDeviceVector(standard_deviation, norm_sum_of_squares);

    // Normalize the values
    tsnecuda::util::BroadcastMatrixVector(d_points, standard_deviation, num_points,
            num_dims, thrust::divides<float>(), 1, 1.f);
}

void tsnecuda::util::SquareDeviceVector(thrust::device_vector<float> &d_out,
        const thrust::device_vector<float> &d_input) {
    thrust::transform(d_input.begin(), d_input.end(),
                      d_out.begin(), tsnecuda::util::FunctionalSquare());
}

void tsnecuda::util::SqrtDeviceVector(thrust::device_vector<float> &d_out,
        const thrust::device_vector<float> &d_input) {
    thrust::transform(d_input.begin(), d_input.end(),
                      d_out.begin(), tsnecuda::util::FunctionalSqrt());
}

float tsnecuda::util::L2NormDeviceVector(
        const thrust::device_vector<float> &d_vector) {
    return std::sqrt(thrust::transform_reduce(d_vector.begin(),
                     d_vector.end(), tsnecuda::util::FunctionalSquare(), 0.0f,
                     thrust::plus<float>()));
}

bool tsnecuda::util::AnyNanOrInfDeviceVector(
        const thrust::device_vector<float> &d_vector) {
    return thrust::transform_reduce(d_vector.begin(), d_vector.end(),
                tsnecuda::util::FunctionalNanOrInf(), 0, thrust::plus<bool>());
}

void tsnecuda::util::MaxNormalizeDeviceVector(
        thrust::device_vector<float> &d_vector) {
    float max_val = thrust::transform_reduce(d_vector.begin(), d_vector.end(),
            tsnecuda::util::FunctionalAbs(), 0.0f, thrust::maximum<float>());
    thrust::constant_iterator<float> division_iterator(max_val);
    thrust::transform(d_vector.begin(), d_vector.end(), division_iterator,
                      d_vector.begin(), thrust::divides<float>());
}

void tsnecuda::util::SymmetrizeMatrix(cusparseHandle_t &handle,
        thrust::device_vector<float> &d_symmetrized_values,
        thrust::device_vector<int32_t> &d_symmetrized_rowptr,
        thrust::device_vector<int32_t> &d_symmetrized_colind,
        thrust::device_vector<float> &d_values,
        thrust::device_vector<int32_t> &d_indices,
        const float magnitude_factor,
        const int num_points, 
        const int num_neighbors) 
{

    // Allocate memory
    int32_t *csr_row_ptr_a = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csr_row_ptr_a),
               (num_points+1)*sizeof(int32_t));
    int32_t *csr_column_ptr_a = thrust::raw_pointer_cast(d_indices.data());
    float *csr_values_a = thrust::raw_pointer_cast(d_values.data());

    // Copy the data
    thrust::device_vector<int> d_vector_memory(csr_row_ptr_a,
            csr_row_ptr_a+num_points+1);
    thrust::sequence(d_vector_memory.begin(), d_vector_memory.end(),
                     0, static_cast<int32_t>(num_neighbors));
    thrust::copy(d_vector_memory.begin(), d_vector_memory.end(), csr_row_ptr_a);
    cudaDeviceSynchronize();

    // Initialize the matrix descriptor
    cusparseMatDescr_t matrix_descriptor;
    cusparseCreateMatDescr(&matrix_descriptor);
    cusparseSetMatType(matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matrix_descriptor, CUSPARSE_INDEX_BASE_ZERO);

    // Sort the matrix properly
    size_t permutation_buffer_byte_size = 0;
    void *permutation_buffer = NULL;
    int32_t *permutation = NULL;

    // step 1: Allocate memory buffer
    cusparseXcsrsort_bufferSizeExt(handle, num_points, num_points,
            num_points*num_neighbors, csr_row_ptr_a,
            csr_column_ptr_a, &permutation_buffer_byte_size);
    cudaDeviceSynchronize();
    cudaMalloc(&permutation_buffer,
               sizeof(char)*permutation_buffer_byte_size);

    // step 2: Setup permutation vector permutation to be the identity
    cudaMalloc(reinterpret_cast<void**>(&permutation),
            sizeof(int32_t)*num_points*num_neighbors);
    cusparseCreateIdentityPermutation(handle, num_points*num_neighbors,
                                      permutation);
    cudaDeviceSynchronize();

    // step 3: Sort CSR format
    cusparseXcsrsort(handle, num_points, num_points,
            num_points*num_neighbors, matrix_descriptor, csr_row_ptr_a,
            csr_column_ptr_a, permutation, permutation_buffer);
    cudaDeviceSynchronize();

    // step 4: Gather sorted csr_values
    float* csr_values_a_sorted = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csr_values_a_sorted),
            (num_points*num_neighbors)*sizeof(float));
    cusparseSgthr(handle, num_points*num_neighbors, csr_values_a,
            csr_values_a_sorted, permutation, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    // Free some memory
    cudaFree(permutation_buffer);
    cudaFree(permutation);
    csr_values_a = csr_values_a_sorted;

    // We need A^T, so we do a csr2csc() call
    int32_t* csc_row_ptr_at = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csc_row_ptr_at),
            (num_points*num_neighbors)*sizeof(int32_t));
    int32_t* csc_column_ptr_at = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csc_column_ptr_at),
            (num_points+1)*sizeof(int32_t));
    float* csc_values_at = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csc_values_at),
            (num_points*num_neighbors)*sizeof(float));

    // Do the transpose operation
    cusparseScsr2csc(handle, num_points, num_points,
                     num_neighbors*num_points, csr_values_a, csr_row_ptr_a,
                     csr_column_ptr_a, csc_values_at, csc_row_ptr_at,
                     csc_column_ptr_at, CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    // Now compute the output size of the matrix
    int32_t base_C, num_nonzeros_C;
    int32_t symmetrized_num_nonzeros = -1;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    d_symmetrized_rowptr.resize(num_points+1);
    cusparseXcsrgeamNnz(handle, num_points, num_points,
            matrix_descriptor, num_points*num_neighbors, csr_row_ptr_a,
                csr_column_ptr_a,
            matrix_descriptor, num_points*num_neighbors, csc_column_ptr_at,
                csc_row_ptr_at,
            matrix_descriptor,
            thrust::raw_pointer_cast(d_symmetrized_rowptr.data()),
            &symmetrized_num_nonzeros);
    cudaDeviceSynchronize();

    // Do some useful checking...
    if (-1 != symmetrized_num_nonzeros) {
        num_nonzeros_C = symmetrized_num_nonzeros;
    } else {
        cudaMemcpy(&num_nonzeros_C,
                thrust::raw_pointer_cast(d_symmetrized_rowptr.data()) +
                num_points, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base_C,
                thrust::raw_pointer_cast(d_symmetrized_rowptr.data()),
                sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Allocate memory for the new summed array
    d_symmetrized_colind.resize(num_nonzeros_C);
    d_symmetrized_values.resize(num_nonzeros_C);

    // Sum the arrays
    float kAlpha = 1.0f / (2.0f * num_points);
    float kBeta = 1.0f / (2.0f * num_points);

    cusparseScsrgeam(handle, num_points, num_points,
            &kAlpha, matrix_descriptor, num_points*num_neighbors,
            csr_values_a, csr_row_ptr_a, csr_column_ptr_a,
            &kBeta, matrix_descriptor, num_points*num_neighbors,
            csc_values_at, csc_column_ptr_at, csc_row_ptr_at,
            matrix_descriptor,
            thrust::raw_pointer_cast(d_symmetrized_values.data()),
            thrust::raw_pointer_cast(d_symmetrized_rowptr.data()),
            thrust::raw_pointer_cast(d_symmetrized_colind.data()));
    cudaDeviceSynchronize();

    // Free the memory we were using...
    cudaFree(csr_values_a);
    cudaFree(csc_values_at);
    cudaFree(csr_row_ptr_a);
    cudaFree(csc_column_ptr_at);
    cudaFree(csc_row_ptr_at);
}

__global__
void tsnecuda::util::Csr2CooKernel(volatile int * __restrict__ coo_indices,
                             const int * __restrict__ pij_row_ptr,
                             const int * __restrict__ pij_col_ind,
                             const int num_points,
                             const int num_nonzero)
{
    register int TID, i, j, start, end;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_nonzero) return;
    start = 0; end = num_points + 1;
    i = (num_points + 1) >> 1;
    while (end - start > 1) {
      j = pij_row_ptr[i];
      end = (j > TID) ? i : end;
      start = (j <= TID) ? i : start;
      i = (start + end) >> 1;
    }
    j = pij_col_ind[TID];
    coo_indices[2*TID] = i;
    coo_indices[2*TID+1] = j;
}

void tsnecuda::util::Csr2Coo(tsnecuda::GpuOptions &gpu_opt,
                             thrust::device_vector<int> &coo_indices,
                             thrust::device_vector<int> &pij_row_ptr,
                             thrust::device_vector<int> &pij_col_ind,
                             const int num_points,
                             const int num_nonzero)
{
    const int num_threads = 1024;
    const int num_blocks = iDivUp(num_nonzero, num_threads);
    
    tsnecuda::util::Csr2CooKernel<<<num_blocks, num_threads>>>(thrust::raw_pointer_cast(coo_indices.data()),
                                                               thrust::raw_pointer_cast(pij_row_ptr.data()),
                                                               thrust::raw_pointer_cast(pij_col_ind.data()),
                                                               num_points, num_nonzero);
    GpuErrorCheck(cudaDeviceSynchronize());
}

