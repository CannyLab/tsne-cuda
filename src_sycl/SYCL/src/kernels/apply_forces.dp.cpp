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
    Apply forces to the points with momentum, exaggeration, etc.
*/

#include <sycl/sycl.hpp>
#include "include/kernels/apply_forces.h"

/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/
// Edited to add momentum, repulsive, attr forces, etc.
void IntegrationKernel(
    volatile float* __restrict__ points,        // num_points * 2
    volatile float* __restrict__ attr_forces,   // num_points * 2
    volatile float* __restrict__ rep_forces,    // num_points * 2
    volatile float* __restrict__ gains,         // num_points * 2
    volatile float* __restrict__ old_forces,    // num_points * 2
    const float eta,
    const float normalization,
    const float momentum,
    const float exaggeration,
    const int   num_points,
    sycl::nd_item<1> item)
{
    int tid;
    float dx, dy, ux, uy, gx, gy;

    tid = item.get_global_id(0);
    if ( tid < num_points) {
        ux = old_forces[tid];
        uy = old_forces[tid + num_points];
        gx = gains[tid];
        gy = gains[tid + num_points];
        dx = exaggeration * attr_forces[tid]              - (rep_forces[tid]              / normalization);
        dy = exaggeration * attr_forces[tid + num_points] - (rep_forces[tid + num_points] / normalization);

        gx = (sycl::signbit(dx) != sycl::signbit(ux)) ? gx + 0.2f : gx * 0.8f;
        gy = (sycl::signbit(dy) != sycl::signbit(uy)) ? gy + 0.2f : gy * 0.8f;
        gx = (gx < 0.01f) ? 0.01f : gx;
        gy = (gy < 0.01f) ? 0.01f : gy;

        ux = momentum * ux - eta * gx * dx;
        uy = momentum * uy - eta * gy * dy;

        points[tid] += ux;
        points[tid + num_points] += uy;

        attr_forces[tid]              = 0.0f;
        attr_forces[tid + num_points] = 0.0f;
        rep_forces[tid]               = 0.0f;
        rep_forces[tid + num_points]  = 0.0f;
        old_forces[tid]               = ux;
        old_forces[tid + num_points]  = uy;
        gains[tid]                    = gx;
        gains[tid + num_points]       = gy;
    }
}

void tsnecuda::ApplyForces(
    float* points,
    float* attr_forces,
    float* rep_forces,
    float* gains,
    float* old_forces,
    const float eta,
    const float normalization,
    const float momentum,
    const float exaggeration,
    const int   num_points,
    sycl::queue& myQueue)
{
    const int WG_SIZE = 32; //1024;
    const int NUM_WGS = iDivUp(num_points, WG_SIZE);

    myQueue.parallel_for(
        sycl::nd_range<1>(NUM_WGS * WG_SIZE, WG_SIZE),
        [=](sycl::nd_item<1> item) {

            IntegrationKernel(
                points,
                attr_forces,
                rep_forces,
                gains,
                old_forces,
                eta,
                normalization,
                momentum,
                exaggeration,
                num_points,
                item);
        }
    );
    myQueue.wait_and_throw();
}
