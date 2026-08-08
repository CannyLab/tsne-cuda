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

// /**
//  * @brief Utilities for reduction across a matrix on an axis
//  * 
//  * @file reduce_utils.h
//  * @author David Chan
//  * @date 2018-04-04
//  * Copyright (c) 2018, Regents of the University of California
//  */

// #ifndef SRC_INCLUDE_UTIL_REDUCE_UTILS_H_
// #define SRC_INCLUDE_UTIL_REDUCE_UTILS_H_

// #include <sycl/sycl.hpp>
// #include "oneapi/mkl/blas.hpp"
// #include "common.h"
// #include "cuda_utils.h"

// namespace tsnecuda {
// namespace utils {

// /**
// * @brief Reduce a matrix by summing then multiplying by alpha along the reduction axis
// * 
// * @param d_matrix The NxM matrix to reduce
// * @param N The number of rows in the matrix
// * @param M The number of columns in the matrix
// * @param alpha The alpha to multiply by
// * @param axis The axis to reduce on (0 = rows, 1 = cols)
// * @param myQueue sycl queue
// * @return float* The reduced vector 
// */
// float* ReduceAlpha(
//     const float* d_matrix,
//     const int N,
//     const int M,
//     float alpha,
//     const int axis,
//     sycl::queue& myQueue);

// /**
// * @brief Reduce a matrix by computing the sum of the reduction axis
// * 
// * @param d_matrix The NxM matrix to reduce
// * @param N The number of rows in the matrix
// * @param M The number of columns in the matrix
// * @param axis The axis to reduce on (0 = rows, 1 = cols)
// * @param myQueue sycl queue
// * @return float* The reduced vector 
// */
// float* ReduceSum(
//     const float* d_matrix,
//     const int N,
//     const int M,
//     const int axis,
//     sycl::queue& myQueue);

// }  // namespace utils
// }  // namespace tsnecuda

// #endif  // SRC_INCLUDE_UTIL_REDUCE_UTILS_H_
