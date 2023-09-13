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
 * @brief 
 * 
 * @file debug_utils.cu
 * @author your name
 * @date 2018-05-05
 * 
 */

#include <sycl/sycl.hpp>
#include "include/utils/debug_utils.h"

template <typename T>
void tsnecuda::debug::PrintArray(
    const T* d_matrix,
    const int N,
    const int M,
    sycl::queue& myQueue) {
    
    T* h_matrix = new T[N * M];
    myQueue.memcpy(h_matrix, d_matrix, N * M * sizeof(T)).wait();
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            std::cout << h_matrix[i + j * N] << " ";
        }
        std::cout << std::endl;
    }
}

template void tsnecuda::debug::PrintArray<float>(
    const float*    d_matrix,
    const int N,
    const int M,
    sycl::queue& myQueue);
template void tsnecuda::debug::PrintArray<int64_t>(
    const int64_t*  d_matrix,
    const int N,
    const int M,
    sycl::queue& myQueue);
template void tsnecuda::debug::PrintArray<int32_t>(
    const int32_t*  d_matrix,
    const int N,
    const int M,
    sycl::queue& myQueue);
template void tsnecuda::debug::PrintArray<uint32_t>(
    const uint32_t* d_matrix,
    const int N,
    const int M,
    sycl::queue& myQueue);

template <typename T>
void tsnecuda::debug::printHostData(T* data_h, const int Nmax)
{
    std::cout << "Host data values:\n";
    for (int i = 0; i < 32; ++i) {
        std::cout << data_h[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void tsnecuda::debug::printDeviceData(T* data_d, const int Nmax, sycl::queue& myQueue)
{
    T* data_h = new T[Nmax];
    myQueue.memcpy(data_h, data_d, Nmax * sizeof(T)).wait();
    std::cout << "Device data values:\n";
    for (int i = 0; i < 32; ++i) {
        std::cout << data_h[i] << " ";
    }
    for (int i = 32*1234; i < 32*1234+32; ++i) {
        std::cout << data_h[i] << " ";
    }
    std::cout << std::endl;
    for (int i = Nmax - 32; i < Nmax; ++i) {
        std::cout << data_h[i] << " ";
    }
    std::cout << std::endl;

    delete[] data_h;
}

template <typename T>
void tsnecuda::debug::writeData(T* data, const int N, std::string filename)
{
    std::ofstream out;
    out.open(filename);
    for (int i = 0; i < N; ++i) {
        out << data[i] << std::endl;
    }
    out.close();
}

template void tsnecuda::debug::printDeviceData<float>(float* data_d, const int Nmax, sycl::queue& myQueue);
template void tsnecuda::debug::printDeviceData<int  >(int*   data_d, const int Nmax, sycl::queue& myQueue);

template void tsnecuda::debug::writeData <int>  (int*  data, const int N, std::string filename);
template void tsnecuda::debug::writeData<float>(float* data, const int N, std::string filename);
template void tsnecuda::debug::writeData<std::complex<float>>(std::complex<float>* data, const int N, std::string filename);
