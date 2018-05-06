/**
 * @brief Utilities for different reductions
 * 
 * @file reduce_utils.cu
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#include "include/util/reduce_utils.h"

// expects matrix of size N x M
thrust::device_vector<float> tsnecuda::util::ReduceAlpha(cublasHandle_t &handle,
        const thrust::device_vector<float> &d_matrix,
        const uint32_t N,
        const uint32_t M,
        float alpha,
        const uint32_t axis) {
    if (axis == 0) {
        thrust::device_vector<float> ones(N, 1.f);
        thrust::device_vector<float> means(M);

        float kBeta = 0.f;
        CublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha,
                thrust::raw_pointer_cast(d_matrix.data()), N,
                thrust::raw_pointer_cast(ones.data()), 1, &kBeta,
                thrust::raw_pointer_cast(means.data()), 1));
        return means;
    } else if (axis == 1) {
        thrust::device_vector<float> ones(M, 1.f);
        thrust::device_vector<float> means(N);

        float kBeta = 0.f;
        CublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha,
                thrust::raw_pointer_cast(d_matrix.data()), N,
                thrust::raw_pointer_cast(ones.data()), 1, &kBeta,
                thrust::raw_pointer_cast(means.data()), 1));
        return means;
    } else {
        throw std::runtime_error("Axis must be 0 or 1.");
    }
}

thrust::device_vector<float> tsnecuda::util::ReduceMean(cublasHandle_t &handle,
        const thrust::device_vector<float> &d_matrix,
        const uint32_t N,
        const uint32_t M,
        const uint32_t axis) {
    float alpha = 1.f / N;
    return tsnecuda::util::ReduceAlpha(handle, d_matrix, N, M, alpha, axis);
}


thrust::device_vector<float> tsnecuda::util::ReduceSum(cublasHandle_t &handle,
        const thrust::device_vector<float> &d_matrix,
        const uint32_t N,
        const uint32_t M,
        const uint32_t axis) {
    float alpha = 1.f;
    return tsnecuda::util::ReduceAlpha(handle, d_matrix, N, M, alpha, axis);
}