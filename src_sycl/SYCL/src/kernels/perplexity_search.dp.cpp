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

/*
    Compute the unnormalized pij matrix given a squared distance matrix and a target perplexity.
    pij = exp(-beta * dist ** 2)

    Note that FAISS returns the first row as the same point, with distance = 0. pii is defined as zero.
*/

#include <sycl/sycl.hpp>
#include "include/kernels/perplexity_search.h"

void ComputePijKernel(
    volatile float* __restrict__ pij,
    const    float* __restrict__ squared_dist,
    const    float* __restrict__ betas,
    const unsigned int num_points,
    const unsigned int num_neighbors,
    sycl::nd_item<1> item)
{
    int tid, i, j;
    float dist, beta;

    tid = item.get_global_id(0);
    if (tid >= num_points * num_neighbors)
        return;

    i = tid / num_neighbors;
    j = tid % num_neighbors;

    beta = betas[i];
    dist = squared_dist[tid];

    // condition deals with evaluation of pii
    // FAISS neighbor zero is i so ignore it
    pij[tid] = (j == 0 & dist == 0.0f) ? 0.0f : sycl::exp(-beta * dist); //TODO: This probably never evaluates to true
}

void RowSumKernel(
    volatile float* __restrict__ row_sum,
    const    float* __restrict__ pij,
    const unsigned int num_points,
    const unsigned int num_neighbors,
    sycl::nd_item<1> item)
{
    int tid = item.get_global_id(0);
    if (tid >= num_points) {
        return;
    }

    float temp_sum = 0.0f;
    for (int j = 0; j < num_neighbors; ++j) {
        temp_sum += pij[tid * num_neighbors + j];
    }
    row_sum[tid] = temp_sum;
}

void NegEntropyKernel(
    volatile float* __restrict__ neg_entropy,
    const    float* __restrict__ pij,
    const unsigned int num_points,
    const unsigned int num_neighbors,
    sycl::nd_item<1> item)
{
    int tid = item.get_global_id(0);
    if (tid >= num_points) {
        return;
    }

    float temp_sum = 0.0f;
    for (int j = 0; j < num_neighbors; ++j) {
        float x = pij[tid * num_neighbors + j];
        temp_sum += (x == 0.0f ? 0.0f : x * sycl::log(x));
    }
    neg_entropy[tid] = -1.0f * temp_sum;
}

void PerplexitySearchKernel(
    volatile float* __restrict__ betas,
    volatile float* __restrict__ lower_bound,
    volatile float* __restrict__ upper_bound,
    volatile int*   __restrict__ found,
    const    float* __restrict__ neg_entropy,
    const    float* __restrict__ row_sum,
    const float perplexity_target,  // 50.0f
    const float epsilon,            // 1e-4
    const int num_points,
    sycl::nd_item<1> item)
{
    int tid, is_found;
    float perplexity, neg_ent, sum_P, perplexity_diff, beta, min_beta, max_beta;

    tid = item.get_global_id(0);
    if (tid >= num_points)
        return;

    neg_ent  = neg_entropy[tid];
    sum_P    = row_sum[tid];
    beta     = betas[tid];
    min_beta = lower_bound[tid];
    max_beta = upper_bound[tid];

    perplexity      = (neg_ent / sum_P) + sycl::log(sum_P);
    perplexity_diff = perplexity - sycl::log((float)perplexity_target);
    is_found        = (perplexity_diff < epsilon && -perplexity_diff < epsilon);
    if (!is_found)
    {
        if (perplexity_diff > 0)
        {
            min_beta = beta;
            beta = (max_beta == FLT_MAX || max_beta == -FLT_MAX) ? beta * 2.0f : (beta + max_beta) / 2.0f;
        }
        else
        {
            max_beta = beta;
            beta = (min_beta == -FLT_MAX || min_beta == FLT_MAX) ? beta / 2.0f : (beta + min_beta) / 2.0f;
        }
        betas[tid] = beta;
        lower_bound[tid] = min_beta;
        upper_bound[tid] = max_beta;
    }
    found[tid] = is_found;
}

void tsnecuda::SearchPerplexity(
    float* pij,                     // output array
    float* squared_dist,            // input array
    const float perplexity_target,
    const float epsilon,
    const int num_points,
    const int num_neighbors,
    sycl::queue& myQueue)
{
    // use beta instead of sigma (this matches the bhtsne code but not the paper)
    // beta is just multiplicative instead of divisive (changes the way binary search works)
    float* betas            = sycl::malloc_device<float>(num_points, myQueue);
    float* lower_bound_beta = sycl::malloc_device<float>(num_points, myQueue);
    float* upper_bound_beta = sycl::malloc_device<float>(num_points, myQueue);
    // float* entropy          = sycl::malloc_device<float>(num_points * num_neighbors, myQueue);
    int* found              = sycl::malloc_device<int>(num_points, myQueue);

    myQueue.fill(betas,                1.0f, num_points);
    myQueue.fill(lower_bound_beta, -FLT_MAX, num_points);
    myQueue.fill(upper_bound_beta,  FLT_MAX, num_points);

    // TODO: this doesn't really fit with the style
    const int WG_SIZE1 = 32; //1024;
    const int NUM_WGS1 = iDivUp(num_points * num_neighbors, WG_SIZE1);

    const int WG_SIZE2 = 32; //128
    const int NUM_WGS2 = iDivUp(num_points, WG_SIZE2);

    size_t iters  = 0;
    int* all_found = sycl::malloc_shared<int>(1, myQueue);

    float* row_sum = sycl::malloc_device<float>(num_points, myQueue);
    float* neg_entropy = sycl::malloc_device<float>(num_points, myQueue);

    do {
        // compute Gaussian Kernel row
        auto e1 =myQueue.parallel_for(
            sycl::nd_range<1>{NUM_WGS1 * WG_SIZE1, WG_SIZE1},
            [=](sycl::nd_item<1> item) {

                ComputePijKernel(
                    pij,                // output
                    squared_dist,
                    betas,
                    num_points,
                    num_neighbors,
                    item);
            }
        );
        // myQueue.wait_and_throw();

        // compute entropy of current row
        auto e2 = myQueue.parallel_for(
            sycl::nd_range<1>{NUM_WGS2 * WG_SIZE2, WG_SIZE2}, e1,
            [=](sycl::nd_item<1> item) {
                
                RowSumKernel(
                    row_sum,            // output
                    pij,
                    num_points,
                    num_neighbors,
                    item);
            }
        );
        // myQueue.wait_and_throw();

        // compute negative entropy
        auto e3 = myQueue.parallel_for(
            sycl::nd_range<1>{NUM_WGS2 * WG_SIZE2, WG_SIZE2}, e1,
            [=](sycl::nd_item<1> item) {
                
                NegEntropyKernel(
                    neg_entropy,        // output
                    pij,
                    num_points,
                    num_neighbors,
                    item);
            }
        );
        // myQueue.wait_and_throw();

        // binary search for beta
        auto e4 = myQueue.parallel_for(
            sycl::nd_range<1>{NUM_WGS2 * WG_SIZE2, WG_SIZE2}, {std::move(e2), std::move(e3)},
            [=](sycl::nd_item<1> item) {
                
                PerplexitySearchKernel(
                    betas,              // output
                    lower_bound_beta,   // output
                    upper_bound_beta,   // output
                    found,              // output
                    neg_entropy,
                    row_sum,
                    perplexity_target,
                    epsilon,
                    num_points,
                    item);
            }
        );
        // myQueue.wait_and_throw();

        // Check if searching is done
        myQueue.parallel_for(
            sycl::nd_range<1>{NUM_WGS2 * WG_SIZE2, WG_SIZE2}, std::move(e4),
            sycl::reduction(all_found, sycl::plus<int>(), sycl::property::reduction::initialize_to_identity{}),
            [=](sycl::nd_item<1> item, auto& sum) {
                int i = item.get_global_id(0);
                if (i >= num_points) return;
                sum += found[i];
            }
        );
        myQueue.wait_and_throw();

        iters++;
    } while ((*all_found) != num_points && iters < 200);
    std::cout << "iters: " << iters << std::endl;
    // TODO: Warn if iters == 200 because perplexity not found?

    tsnecuda::utils::BroadcastMatrixVector(pij, row_sum, num_neighbors, num_points, std::divides<float>(), 1, 1.0f, myQueue);
}
