/**
 * @brief Utilities for different reductions
 * 
 * @file reduce_utils.cu
 * @author David Chan
 * @date 2018-04-04
 */

 #include "util/reduce_utils.h"

// expects matrix of size N x M
thrust::device_vector<float> Reduce::reduce_alpha(cublasHandle_t &handle, 
                                          const thrust::device_vector<float> &matrix, 
                                          const unsigned int N, 
                                          const unsigned int M, 
                                          float alpha, 
                                          const int axis)
{
    if (axis == 0) {
        thrust::device_vector<float> ones(N, 1.f);
        thrust::device_vector<float> means(M);

        float beta = 0.f;
        cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, thrust::raw_pointer_cast(matrix.data()), N,
                                    thrust::raw_pointer_cast(ones.data()), 1, &beta, thrust::raw_pointer_cast(means.data()), 1));
        return means;
    } else if (axis == 1) {
        thrust::device_vector<float> ones(M, 1.f);
        thrust::device_vector<float> means(N);

        float beta = 0.f;
        cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, thrust::raw_pointer_cast(matrix.data()), N,
                                    thrust::raw_pointer_cast(ones.data()), 1, &beta, thrust::raw_pointer_cast(means.data()), 1));
        return means;
    } else {
        throw std::runtime_error("Axis must be 0 or 1.");
    }
}

thrust::device_vector<float> Reduce::reduce_mean(cublasHandle_t &handle, 
                                         const thrust::device_vector<float> &matrix, 
                                         const unsigned int N, 
                                         const unsigned int M, 
                                         const int axis) 
{
    float alpha = 1.f / N;
    return Reduce::reduce_alpha(handle, matrix, N, M, alpha, axis);
}


thrust::device_vector<float> Reduce::reduce_sum(cublasHandle_t &handle, 
                                        const thrust::device_vector<float> &matrix, 
                                        const unsigned int N, 
                                        const unsigned int M, 
                                        const int axis) 
{
    float alpha = 1.f;
    return Reduce::reduce_alpha(handle, matrix, N, M, alpha, axis);
}