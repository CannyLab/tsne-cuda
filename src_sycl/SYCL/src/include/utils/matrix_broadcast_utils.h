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
 * @brief Utilities for broadcasting across a GPU matrix
 * 
 * @file matrix_broadcast.h
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_MATRIX_BROADCAST_UTILS_H_
#define SRC_INCLUDE_UTIL_MATRIX_BROADCAST_UTILS_H_

#include <sycl/sycl.hpp>
#include "common.h"
#include "cuda_utils.h"

namespace tsnecuda {
namespace utils {

/// @private
template<typename BinaryFunction, typename T>
void BroadcastRowVector(
    T*       __restrict__ matrix,
    const T* __restrict__ vector,
    const int N,
    const int M,
    BinaryFunction binary_operation,
    const T alpha,
    sycl::nd_item<1> item_ct1);

/// @private
template<typename BinaryFunction, typename T>
void BroadcastColumnVector(
    T*       __restrict__ matrix,
    const T* __restrict__ vector,
    const int N,
    const int M,
    BinaryFunction binary_operation,
    const T alpha,
    sycl::nd_item<1> item_ct1);

/**
* @brief Function for broadcasting a vector across a matrix
* 
* @tparam BinaryFunction The function to broadcast with
* @tparam T Matrix format
* @param matrix (N x M) matrix stored in column major order
* @param vector Length N vector if axis == 0, length M vector if axis == 1
* @param N,M dimensions of matrix
* @param binary_operation an operation that takes in two arguments of type T and returns a type T
* @param axis 0 or 1, controlls whether this runs a column or row broadcast
* @param alpha scalar multiple for vector
* 
* @note 
* should axis == 0 be row or column broadcasting? and vice versa for axis == 1?
*/
template<typename BinaryFunction, typename T>
void BroadcastMatrixVector(
    T* d_matrix,
    const T* d_vector,
    const int N,
    const int M,
    BinaryFunction binary_operation,
    const int axis,
    const T alpha,
    sycl::queue& myQueue);
}  // namespace utils
}  // namespace tsnecuda

#endif  // SRC_INCLUDE_UTIL_MATRIX_BROADCAST_UTILS_H_
