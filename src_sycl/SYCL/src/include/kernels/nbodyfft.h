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

#ifndef NBODYFFT_H
#define NBODYFFT_H

#include <sycl/sycl.hpp>
#include <complex>

#include "common.h"
#include "utils/cuda_utils.h"
#include "utils/debug_utils.h"
#include "utils/matrix_broadcast_utils.h"

// #define atomicAddF(x, y) (sycl::ext::oneapi::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::global_space>(*(x)) +=(y))
#define atomicAdd(x, y) (sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::global_space>(*(x)) +=(y))

namespace tsnecuda
{

void PrecomputeFFT2D(
    // std::shared_ptr<descriptor_t>& plan_kernel_tilde,
    float x_max,
    float x_min,
    float y_max,
    float y_min,
    int n_boxes,
    int n_interpolation_points,
    float* box_lower_bounds_device,
    float* box_upper_bounds_device,
    float* kernel_tilde_device,
    std::complex<float>* fft_kernel_tilde_device,
    std::complex<float>* fft_scratchpad_device,
    sycl::queue& myQueue, double& duration);

void NbodyFFT2D(
    // std::shared_ptr<descriptor_t>& plan_dft,
    // std::shared_ptr<descriptor_t>& plan_idft,
    std::complex<float>* fft_kernel_tilde_device,
    std::complex<float>* fft_w_coefficients,
    int    N,
    int    n_terms,
    int    n_boxes,
    int    n_interpolation_points,
    int    n_total_boxes,
    int    total_interpolation_points,
    float  coord_min,
    float  box_width,
    int    n_fft_coeffs_half,
    int    n_fft_coeffs,
    float* fft_input,
    float* fft_output,
    int*   point_box_idx_device,
    float* x_in_box_device,
    float* y_in_box_device,
    float* points_device,
    float* box_lower_bounds_device,
    float* y_tilde_spacings_device,
    float* denominator_device,
    float* y_tilde_values,
    // float* all_interpolated_values_device,
    // float* output_values,
    // int*   all_interpolated_indices,
    // int*   output_indices,
    float* w_coefficients_device,
    float* chargesQij_device,
    float* x_interpolated_values_device,
    float* y_interpolated_values_device,
    float* potentialsQij_device,
    std::complex<float>* fft_scratchpad_device,
    sycl::queue& myQueue, double& duration);

} // namespace tsnecuda

#endif
