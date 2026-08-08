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
 * @brief Implementation of the math_utils.h file
 *
 * @file math_utils.cu
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#include <oneapi/dpl/numeric>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include "include/utils/math_utils.h"

float tsnecuda::utils::L2NormDeviceVector(
    const float* d_vector,
    const int N,
    sycl::queue& myQueue)
{
    return std::sqrt(std::transform_reduce(
        oneapi::dpl::execution::make_device_policy(myQueue),
        d_vector,
        d_vector + N,
        0.0f,
        std::plus<float>(),
        tsnecuda::utils::FunctionalSquare()));
}

void tsnecuda::utils::MaxNormalizeDeviceVector(
    float* d_vector,
    const int N,
    sycl::queue& myQueue)
{
    auto policy = oneapi::dpl::execution::make_device_policy(myQueue);

    float max_val = std::transform_reduce(
        policy,
        d_vector,
        d_vector + N,
        0.0f,
        oneapi::dpl::maximum<float>(),
        tsnecuda::utils::FunctionalAbs());  // works w/ correct includes

    float* division_iterator = sycl::malloc_device<float>(N, myQueue);
    myQueue.fill(division_iterator, max_val, N).wait();
    std::transform(
        policy,
        d_vector,
        d_vector + N,
        division_iterator,
        d_vector,
        std::divides<float>());             // works
    
    sycl::free(division_iterator, myQueue);
}

void syv2k(
          float* __restrict__ pij_sym,
    const float* __restrict__ pij_non_sym,
    const int*   __restrict__ pij_indices,
    const int num_points,
    const int num_neighbors,
    sycl::nd_item<1> item)
{
    int tid, i, j, jend;
    float pij_acc;

    tid = item.get_global_id(0);
    if (tid >= (num_points * num_neighbors)) {
        return;
    }

    i = tid / num_neighbors;
    j = pij_indices[tid];

    pij_acc = pij_non_sym[tid];
    jend = (j + 1) * num_neighbors;
    for (int jidx = j * num_neighbors; jidx < jend; jidx++) {
        pij_acc += pij_indices[jidx] == i ? pij_non_sym[jidx] : 0.0f;
    }
    pij_sym[tid] = pij_acc / (2.0f * num_points);
}

void tsnecuda::utils::SymmetrizeMatrixV2(
          float*   pij_symmetric,     // output
    const float*   pij_nonsymmetric,  // input
    const int32_t* pij_indices,       // input
    const int num_points,
    const int num_neighbors,
    sycl::queue& myQueue)
{
    const int wg_size = 32; //1024;
    const int num_wgs = iDivUp(num_points * num_neighbors, wg_size);

    myQueue.parallel_for(
        sycl::nd_range<1>{num_wgs * wg_size, wg_size},
        [=](sycl::nd_item<1> item) {

            syv2k(
                pij_symmetric,
                pij_nonsymmetric,
                pij_indices,
                num_points,
                num_neighbors,
                item);
        }
    );
    myQueue.wait_and_throw();
}
