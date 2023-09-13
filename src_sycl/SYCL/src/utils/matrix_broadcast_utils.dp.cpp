/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @brief Implementation file for matrix broadcasting
 * 
 * @file matrix_broadcast.cu
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#include <sycl/sycl.hpp>
#include "include/utils/matrix_broadcast_utils.h"

// Performs the operation matrix[i, :] = binary_op(matrix[i, :],
// alpha * vector) for each row i in the matrix
template<typename BinaryFunction, typename T>
void tsnecuda::utils::BroadcastRowVector(
          T* __restrict__ d_matrix,
    const T* __restrict__ d_vector,
    const int N,
    const int M,
    BinaryFunction binary_operation,
    const T alpha,
    sycl::nd_item<1> item)
{
    const int tid = item.get_global_id(0);
    const int i = tid % N;
    const int j = tid / N;
    if (j < M) {
        d_matrix[j * N + i] = binary_operation(d_matrix[j * N + i], alpha * d_vector[j]);
    }
}

// Performs the operation matrix[:, j] = binary_op(matrix[:, j],
// alpha * vector) for each col i in the matrix
template<typename BinaryFunction, typename T>
void tsnecuda::utils::BroadcastColumnVector(
          T* __restrict__ d_matrix,     // 4 x 780 x (780 / 2 + 1) = 4 x 304980 = 1219920
    const T* __restrict__ d_vector,     // 780 x 780 = 608400
    const int N,                        // 780 x (780 / 2 + 1) = 304980
    const int M,                        // 4
    BinaryFunction binary_operation,
    const T alpha,
    sycl::nd_item<1> item)
{
    const int tid = item.get_global_id(0);
    const int i = tid % N;
    const int j = tid / N;

    if (j < M) {    // condition makes sure tid < size of d_matrix
        d_matrix[j * N + i] = binary_operation(d_matrix[j * N + i], alpha * d_vector[i]);
    }
}

template<typename BinaryFunction, typename T>
void tsnecuda::utils::BroadcastMatrixVector(
          T* d_matrix,
    const T* d_vector,
    const int N,
    const int M,
    BinaryFunction binary_operation,
    const int axis,
    const T alpha,
    sycl::queue& myQueue)
{
    // // Checks to make sure dimensions are correct
    // assert(d_matrix.size() >= N * M);
    // assert((axis == 0 && d_vector.size() >= N) ||
    //        (axis == 1 && d_vector.size() >= M));

    const int kBlockSize = 32;
    const int kNumBlocks = iDivUp(N * M, kBlockSize);

    // if (axis == 0) {
        myQueue.parallel_for(
            sycl::nd_range<1>(kNumBlocks * kBlockSize, kBlockSize),
            [=](sycl::nd_item<1> item) {

                BroadcastColumnVector(
                    d_matrix,
                    d_vector,
                    N,
                    M,
                    binary_operation,
                    alpha,
                    item);
            }
        );
    // } else {
    //     myQueue.parallel_for(
    //         sycl::nd_range<1>(kNumBlocks * kBlockSize, kBlockSize),
    //         [=](sycl::nd_item<1> item) {

    //             BroadcastRowVector(
    //                 d_matrix,
    //                 d_vector,
    //                 N,
    //                 M,
    //                 binary_operation,
    //                 alpha,
    //                 item);
    //         }
    //     );
    // }
    myQueue.wait_and_throw();
}

// Explicit instantiations of the method
template void tsnecuda::utils::BroadcastMatrixVector<std::divides<float>, float>(
          float* d_matrix,
    const float* d_vector,
    const int N,
    const int M,
    std::divides<float> binary_operation,
    const int axis,
    const float alpha,
    sycl::queue& myQueue);

template void tsnecuda::utils::BroadcastMatrixVector<std::minus<float>, float>(
          float* d_matrix,
    const float* d_vector,
    const int N,
    const int M,
    std::minus<float> binary_operation,
    const int axis,
    const float alpha,
    sycl::queue& myQueue);

template void tsnecuda::utils::BroadcastMatrixVector<std::multiplies<std::complex<float>>, std::complex<float>>(
          std::complex<float>* d_matrix,
    const std::complex<float>* d_vector,
    const int N,
    const int M,
    std::multiplies<std::complex<float>> binary_operation,
    const int axis,
    const std::complex<float> alpha,
    sycl::queue& myQueue);
