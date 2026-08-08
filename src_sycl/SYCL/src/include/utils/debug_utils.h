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
 * @brief Debugging Utilities
 * 
 * @file debug_utils.h
 * @author David chan
 * @date 2018-05-05
 * Copyright (c) 2018, Regents of the University of California
 */


#ifndef SRC_INCLUDE_UTIL_DEBUG_UTILS_H_
#define SRC_INCLUDE_UTIL_DEBUG_UTILS_H_

#include <sycl/sycl.hpp>
#include "common.h"

namespace tsnecuda {
namespace debug {

/**
 * @brief Print the NxM device matrix
 * 
 * @tparam T The type of the device matrix
 * @param d_matrix The NxM matrix to print
 * @param N The number of rows in the matrix
 * @param M The number of columns in the matrix
 */
template <typename T>
void PrintArray(
    const T* d_matrix,
    const int N,
    const int M,
    sycl::queue& myQueue);

template <typename T>
void printHostData(T* data_h, const int Nmax);

template <typename T>
void printDeviceData(T* data_d, const int Nmax, sycl::queue& myQueue);

template <typename T>
void writeData(T* data, const int N, std::string filename);

}  // namespace debug
}  // namespace tsnecuda


#endif  // SRC_INCLUDE_UTIL_DEBUG_UTILS_H_
