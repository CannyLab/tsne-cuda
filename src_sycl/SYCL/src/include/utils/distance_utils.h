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
 * @brief Utilities for the computation of various distances
 *
 * @file distance_utils.h
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_DISTANCE_UTILS_H_
#define SRC_INCLUDE_UTIL_DISTANCE_UTILS_H_

// CXX Includes
#include <sycl/sycl.hpp>
#include <stdint.h>

// Local Includes
#include "common.h"
#include "options.h"
#include "cuda_utils.h"
// #include "reduce_utils.h"
#include "math_utils.h"

namespace tsnecuda
{
namespace utils
{

/**
* @brief Use FAISS to compute the k-nearest neighbors for the given points
*
* @param gpu_opt GPU Options object
* @param base_opt Base Options object
* @param indices The index array that goes with the distance array (N_POINTSxK) row-major (so I[K*i + j] gives the j'th nearest neighbor of the i'th point)
* @param distances The euclidean distance array (true euclidean distance, not squared) (N_POINTSxK) row-major
* @param points The points of which you want the k nearest neighbors (N_POINTSxN_DIMS) row-major (so points[N_DIM*i + j] gives the j'th dim of the i'th point)
* @param num_dims The number of dimensions of the input points
* @param num_points The number of input points
* @param K The number of nearest neighbors to return. If >=1024, this function uses the CPU instead of the GPU
*/
void KNearestNeighbors(
    std::string data_folder,
    int64_t* indices,
    float* distances,
    const int num_dims,
    const int num_points,
    const int num_near_neighbots);

void PostprocessNeighborIndices(
    int* pij_indices,
    int64_t* knn_indices,
    const int num_points,
    const int num_neighbors,
    sycl::queue& myQueue);
} // namespace utils
} // namespace tsnecuda
#endif // SRC_INCLUDE_UTIL_DISTANCE_UTILS_H_
