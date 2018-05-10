/**
 * @brief 
 * 
 * @file debug_utils.cu
 * @author your name
 * @date 2018-05-05
 * 
 */

#include "include/util/debug_utils.h"

template <typename T>
void tsnecuda::debug::PrintArray(const thrust::device_vector<T> &d_matrix,
        const int N, const int M) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            std::cout << d_matrix[i + j * N] << " ";
        }
        std::cout << std::endl;
    }
}

template void tsnecuda::debug::PrintArray<float>(
        const thrust::device_vector<float> &d_matrix,
        const int N, const int M);
template void tsnecuda::debug::PrintArray<int64_t>(
        const thrust::device_vector<int64_t> &d_matrix,
        const int N, const int M);
template void tsnecuda::debug::PrintArray<int32_t>(
        const thrust::device_vector<int32_t> &d_matrix,
        const int N, const int M);
template void tsnecuda::debug::PrintArray<uint32_t>(
        const thrust::device_vector<uint32_t> &d_matrix,
        const int N, const int M);
