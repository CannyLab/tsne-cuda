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

#include <sycl/sycl.hpp>
#include "../include/kernels/atomic.h"
#include <complex>
#include "../include/kernels/nbodyfft.h"

const float PI = 3.14159265358979f;
// const float twoPI = 3.14159265358979f * 2;


// #define BS1 1
#define BS1 16

#define BS2 16
// #define BS2 32
// #define BS2 1024

#define TWIDDLE()                                   \
    sinf = sycl::sin(angle * k);                    \
    cosf = sycl::cos(angle * k);                    \
    twiddle = std::complex<float>(cosf, sinf);
/*
#define TWIDDLE()                                                                                               \
    sinf = sycl::sincos(angle * k, sycl::make_ptr<float, sycl::access::address_space::private_space>(&cosf));   \
    twiddle = std::complex<float>(cosf, sinf);
*/

void copy_to_fft_input(
    volatile float* __restrict__ fft_input,
    const    float* w_coefficients_device,
    const int n_fft_coeffs,
    const int n_fft_coeffs_half,
    const int n_terms,
    sycl::nd_item<1> item)
{
    int i, j;
    int tid = item.get_global_id(0);
    if (tid >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
        return;

    int current_term = tid / (n_fft_coeffs_half * n_fft_coeffs_half);
    int current_loc  = tid % (n_fft_coeffs_half * n_fft_coeffs_half);

    i = current_loc / n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half;

    fft_input[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] = w_coefficients_device[current_term + current_loc * n_terms];
}

void copy_from_fft_output(
    volatile float* __restrict__ y_tilde_values,
    const    float* fft_output,                     // n_terms * n_fft_coeffs *  n_fft_coeffs -> n_terms * n_fft_coeffs * (n_fft_coeffs + 2)
    const int n_fft_coeffs,
    const int n_fft_coeffs_half,
    const int n_terms,
    sycl::nd_item<1> item)
{
    int i, j;
    int tid = item.get_global_id(0);
    if (tid >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
        return;

    int current_term = tid / (n_fft_coeffs_half * n_fft_coeffs_half);
    int current_loc  = tid % (n_fft_coeffs_half * n_fft_coeffs_half);

    i = current_loc / n_fft_coeffs_half + n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half + n_fft_coeffs_half;

    y_tilde_values[current_term + n_terms * current_loc] = fft_output[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] / (float)(n_fft_coeffs * n_fft_coeffs);
    // y_tilde_values[current_term + n_terms * current_loc] = fft_output[current_term * (n_fft_coeffs * (n_fft_coeffs+2)) + i * (n_fft_coeffs+2) + j] / (float)(n_fft_coeffs * n_fft_coeffs);
}

void compute_point_box_idx(
    volatile int*   __restrict__ point_box_idx,
    volatile float* __restrict__ x_in_box,
    volatile float* __restrict__ y_in_box,
    const float* const xs,
    const float* const ys,
    const float* const box_lower_bounds,
    const float min_coord,
    const float box_width,
    const int n_boxes,
    const int n_total_boxes,
    const int N,
    sycl::nd_item<1> item)
{
    int tid = item.get_global_id(0);
    if (tid >= N)
        return;

    int x_idx = (int)((xs[tid] - min_coord) / box_width);
    int y_idx = (int)((ys[tid] - min_coord) / box_width);

    x_idx = sycl::max(0, x_idx);
    x_idx = sycl::min((int)(n_boxes - 1), x_idx);

    y_idx = sycl::max(0, y_idx);
    y_idx = sycl::min((int)(n_boxes - 1), y_idx);

    int box_idx = y_idx * n_boxes + x_idx;
    point_box_idx[tid] = box_idx;

    x_in_box[tid] = (xs[tid] - box_lower_bounds[box_idx])                 / box_width;
    y_in_box[tid] = (ys[tid] - box_lower_bounds[n_total_boxes + box_idx]) / box_width;
}

void interpolate_device(
    volatile float* __restrict__ interpolated_values,
    const    float* const y_in_box,
    const    float* const y_tilde_spacings,
    const    float* const denominator,
    const int n_interpolation_points,
    const int N,
    sycl::nd_item<1> item)
{
    int tid, i, j, k;
    float value, ybox_i;

    tid = item.get_global_id(0);
    if (tid >= N * n_interpolation_points)
        return;

    i = tid % N;
    j = tid / N;

    value = 1;
    ybox_i = y_in_box[i];

    for (k = 0; k < n_interpolation_points; k++) {
        if (j != k) {
            value *= ybox_i - y_tilde_spacings[k];
        }
    }

    interpolated_values[j * N + i] = value / denominator[j];
}

void compute_interpolated_indices(
          float* __restrict__ w_coefficients_device,
    const int*   const point_box_indices,
    const float* const chargesQij,
    const float* const x_interpolated_values,
    const float* const y_interpolated_values,
    const int N,
    const int n_interpolation_points,
    const int n_boxes,
    const int n_terms,
    sycl::nd_item<1> item)
{
    int tid, current_term, i, interp_i, interp_j, box_idx, box_i, box_j, idx;
    tid = item.get_global_id(0);
    if (tid >= n_terms * n_interpolation_points * n_interpolation_points * N)
        return;

    current_term = tid % n_terms;
    i = (tid / n_terms) % N;
    interp_j = ((tid / n_terms) / N) % n_interpolation_points;
    interp_i = ((tid / n_terms) / N) / n_interpolation_points;

    box_idx = point_box_indices[i];
    box_i = box_idx % n_boxes;
    box_j = box_idx / n_boxes;

    idx = (box_i * n_interpolation_points  + interp_i) * (n_boxes * n_interpolation_points) +
          (box_j * n_interpolation_points) + interp_j;

    atomicAdd(
        w_coefficients_device + idx * n_terms + current_term,
        x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * chargesQij[i * n_terms + current_term]);
}

void compute_potential_indices(
          float* __restrict__ potentialsQij,
    const int*   const point_box_indices,
    const float* const y_tilde_values,
    const float* const x_interpolated_values,
    const float* const y_interpolated_values,
    const int N,
    const int n_interpolation_points,
    const int n_boxes,
    const int n_terms,
    sycl::nd_item<1> item)
{
    int tid, current_term, i, interp_i, interp_j, box_idx, box_i, box_j, idx;
    tid = item.get_global_id(0);
    if (tid >= n_terms * n_interpolation_points * n_interpolation_points * N)
        return;

    current_term = tid % n_terms;
    i = (tid / n_terms) % N;
    interp_j = ((tid / n_terms) / N) % n_interpolation_points;
    interp_i = ((tid / n_terms) / N) / n_interpolation_points;

    box_idx = point_box_indices[i];
    box_i = box_idx % n_boxes;
    box_j = box_idx / n_boxes;

    idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
          (box_j * n_interpolation_points) + interp_j;

    atomicAdd(
        potentialsQij + i * n_terms + current_term,
        x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * y_tilde_values[idx * n_terms + current_term]);
}

float squared_cauchy_2d(float x1, float x2, float y1, float y2)
{
    return sycl::pown(1.0f + (x1 - y1) * (x1 - y1) + (x2 - y2) * (x2 - y2), -2);
}

void compute_kernel_tilde(
    volatile float* __restrict__ kernel_tilde,   // 780 x 780
    const    float x_min,
    const    float y_min,
    const    float h,
    const    int   n_interpolation_points_1d,    // 390
    const    int   n_fft_coeffs,                 // 390 x 2 = 780
    sycl::nd_item<1> item)
{
    int tid, i, j;
    float tmp;
    tid = item.get_global_id(0);
    if (tid >= n_interpolation_points_1d * n_interpolation_points_1d)
        return;

    i = tid / n_interpolation_points_1d;
    j = tid % n_interpolation_points_1d;

    // TODO: Possibly issuing a memory pre-fetch here could help the code.
    tmp = squared_cauchy_2d(y_min + h / 2, x_min + h / 2, y_min + h / 2 + i * h, x_min + h / 2 + j * h);
    kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
    // kernel_tilde[(n_interpolation_points_1d + i) * ((n_fft_coeffs/2+1)*2) + (n_interpolation_points_1d + j)] = tmp;
    // kernel_tilde[(n_interpolation_points_1d - i) * ((n_fft_coeffs/2+1)*2) + (n_interpolation_points_1d + j)] = tmp;
    // kernel_tilde[(n_interpolation_points_1d + i) * ((n_fft_coeffs/2+1)*2) + (n_interpolation_points_1d - j)] = tmp;
    // kernel_tilde[(n_interpolation_points_1d - i) * ((n_fft_coeffs/2+1)*2) + (n_interpolation_points_1d - j)] = tmp;
}

void compute_upper_and_lower_bounds(
    volatile float* __restrict__ box_upper_bounds,
    volatile float* __restrict__ box_lower_bounds,
    const    float box_width,
    const    float x_min,
    const    float y_min,
    const    int   n_boxes,
    const    int   n_total_boxes,
    sycl::nd_item<1> item)
{
    int tid, i, j;
    tid = item.get_global_id(0);
    if (tid >= n_boxes * n_boxes)
        return;

    i = tid / n_boxes;
    j = tid % n_boxes;

    box_lower_bounds[i * n_boxes + j] =  j      * box_width + x_min;
    box_upper_bounds[i * n_boxes + j] = (j + 1) * box_width + x_min;

    box_lower_bounds[n_total_boxes + i * n_boxes + j] =  i      * box_width + y_min;
    box_upper_bounds[n_total_boxes + i * n_boxes + j] = (i + 1) * box_width + y_min;
}

// real to complex
void DFT2D1gpu(float* din, std::complex<float>* dout, int num_rows, int num_cols, sycl::nd_item<2> item)
{
	int j = item.get_global_id(0);
    int i = item.get_global_id(1);
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, cosf, sinf; 
    std::complex<float> sum, twiddle;
    angle = -2.0f * PI * ((float)i / (float)num_cols);
    sum = 0.0f;
#pragma unroll
    for (int k = 0; k < num_cols; ++k) {
        // sinf = sycl::sin(angle * k);
        // cosf = sycl::cos(angle * k);
        // sinf = sycl::sincos(angle * k, sycl::make_ptr<float, sycl::access::address_space::private_space>(&cosf));
        // twiddle = std::complex<float>(cosf, sinf);
        TWIDDLE();
        sum = sum + din[j * num_cols + k] * twiddle;
    }

    dout[i * num_rows + j] = sum;
}

// complex to complex
void DFT2D2gpu(std::complex<float>* din, std::complex<float>* dout, int num_rows, int num_cols, sycl::nd_item<2> item)
{
	int j = item.get_global_id(0);
    int i = item.get_global_id(1);
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, cosf, sinf;
    std::complex<float> sum, twiddle;
    angle = -2.0f * PI * ((float)i / (float)num_cols);
    sum = 0.0f;
#pragma unroll
    for (int k = 0; k < num_cols; ++k) {
        // sinf = sycl::sin(angle * k);
        // cosf = sycl::cos(angle * k);
        // sinf = sycl::sincos(angle * k, sycl::make_ptr<float, sycl::access::address_space::private_space>(&cosf));
        // twiddle = std::complex<float>(cosf, sinf);
        TWIDDLE();
        sum = sum + din[j * num_cols + k] * twiddle;
    }

    dout[i * num_rows + j] = sum;
}

// complex to complex
void iDFT2D1gpu(std::complex<float>* din, std::complex<float>* dout, int num_rows, int num_cols, sycl::nd_item<2> item)
{
	int j = item.get_global_id(0);
    int i = item.get_global_id(1);
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, cosf, sinf; 
    std::complex<float> sum, twiddle;
    angle = 2.0f * PI * ((float)i / (float)num_cols);
    sum = 0.0f;
#pragma unroll
    for (int k = 0; k < num_cols; ++k) {
        // sinf = sycl::sin(angle * k);
        // cosf = sycl::cos(angle * k);
        // sinf = sycl::sincos(angle * k, sycl::make_ptr<float, sycl::access::address_space::private_space>(&cosf));
        // twiddle = std::complex<float>(cosf, sinf);
        TWIDDLE();
        if (k < (num_cols/2+1)) {
            sum = sum + din[j * (num_cols/2+1) + k] * twiddle;
        } else {
            sum = sum + std::conj(din[((num_rows-j)%num_rows) * (num_cols/2+1) + ((num_cols-k)%num_cols)]) * twiddle;
        }
    }

    dout[i * num_rows + j] = sum;
}

// complex to real
void iDFT2D2gpu(std::complex<float>* din, float* dout, int num_rows, int num_cols, sycl::nd_item<2> item)
{
	int j = item.get_global_id(0);
    int i = item.get_global_id(1);
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, sum, cosf, sinf;
    std::complex<float> twiddle;
    angle = 2.0f * PI * ((float)i / (float)num_cols);
    sum = 0.0f;
#pragma unroll
    for (int k = 0; k < num_cols; ++k) {
        // sinf = sycl::sin(angle * k);
        // cosf = sycl::cos(angle * k);
        // sinf = sycl::sincos(angle * k, sycl::make_ptr<float, sycl::access::address_space::private_space>(&cosf));
        // twiddle = std::complex<float>(cosf, sinf);
        TWIDDLE();
        sum = sum + (din[j * num_cols + k] * twiddle).real();
    }

    dout[i * num_rows + j] = sum;
}

void tsnecuda::PrecomputeFFT2D(
    // std::shared_ptr<descriptor_t>& plan_tilde,
    float  x_max,
    float  x_min,
    float  y_max,
    float  y_min,
    int    n_boxes,
    int    n_interpolation_points,
    float* box_lower_bounds_device,
    float* box_upper_bounds_device,
    float* kernel_tilde_device,
    std::complex<float>* fft_kernel_tilde_device,
    std::complex<float>* fft_scratchpad_device,
    sycl::queue& myQueue, double& duration)
{
    const int num_threads = 32;
    int num_blocks = (n_boxes * n_boxes + num_threads - 1) / num_threads;
    /*
     * Set up the boxes
     */
    int n_total_boxes = n_boxes * n_boxes;
    float box_width   = (x_max - x_min) / (float)n_boxes;

    // Left and right bounds of each box, first the lower bounds in the x direction, then in the y direction
    myQueue.parallel_for<class compute_upper_and_lower_bounds1>(
        sycl::nd_range<1>(num_blocks * num_threads, num_threads),
        [=](sycl::nd_item<1> item) {

            compute_upper_and_lower_bounds(
                box_upper_bounds_device,    // output
                box_lower_bounds_device,    // output
                box_width,
                x_min,
                y_min,
                n_boxes,
                n_total_boxes,
                item
            );
        }
    );  // wait() not needed as this and next kernels can run simultaneously

    // Coordinates of all the equispaced interpolation points
    int n_interpolation_points_1d = n_interpolation_points    * n_boxes;
    int n_fft_coeffs              = n_interpolation_points_1d * 2;

    float h = box_width / (float)n_interpolation_points;

    /*
     * Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant
     * matrix
     */
    num_blocks = (n_interpolation_points_1d * n_interpolation_points_1d + num_threads - 1) / num_threads;
    auto e1 = myQueue.parallel_for<class compute_kernel_tilde1>(
        sycl::nd_range<1>(num_blocks * num_threads, num_threads),
        [=](sycl::nd_item<1> item) {

            compute_kernel_tilde(
                kernel_tilde_device,        // outuput
                x_min,
                y_min,
                h,
                n_interpolation_points_1d,
                n_fft_coeffs,
                item
            );
        }
    );
    // myQueue.wait_and_throw();

    // // Precompute the FFT of the kernel generating matrix
    // oneapi::mkl::dft::compute_forward(
    //     *plan_tilde,
    //     reinterpret_cast<float *>(kernel_tilde_device),
    //     (float *)reinterpret_cast<sycl::float2 *>(fft_kernel_tilde_device)).wait();

    int num_rows = n_interpolation_points * n_boxes * 2;
    int num_cols = num_rows;

    sycl::range<2> block_size(BS1, BS2);
    sycl::range<2> grid_size1((num_rows       + block_size[0] - 1) / block_size[0], (num_cols + block_size[1] - 1) / block_size[1]);
    sycl::range<2> grid_size2(((num_cols/2+1) + block_size[0] - 1) / block_size[0], (num_rows + block_size[1] - 1) / block_size[1]);

    auto e2 = myQueue.parallel_for<class DFT2D1gpu1>(
        sycl::nd_range<2>(grid_size1 * block_size, block_size), std::move(e1),
        [=](sycl::nd_item<2> item) {

            DFT2D1gpu(
                kernel_tilde_device,
                fft_scratchpad_device,
                num_rows,
                num_cols,
                item
            );
        }
    );
    // myQueue.wait_and_throw();

    myQueue.parallel_for<class DFT2D2gpu1>(
        sycl::nd_range<2>(grid_size2 * block_size, block_size), std::move(e2),
        [=](sycl::nd_item<2> item) {

            DFT2D2gpu(
                fft_scratchpad_device,
                fft_kernel_tilde_device,
                (num_cols/2+1),
                num_rows,
                item
            );
        }
    );
    myQueue.wait_and_throw();
}

void tsnecuda::NbodyFFT2D(
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
    float  min_coord,
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
    sycl::queue& myQueue, double& duration)
{
    const int num_threads = 128;
    int num_blocks = (N + num_threads - 1) / num_threads;

    // Compute box indices and the relative position of each point in its box in the interval [0, 1]
    auto e1 = myQueue.parallel_for<class compute_point_box_idx2>(
        sycl::nd_range<1>(num_blocks * num_threads, num_threads),
        [=](sycl::nd_item<1> item) {

            compute_point_box_idx(
                point_box_idx_device,       // output: matched
                x_in_box_device,            // output: matched
                y_in_box_device,            // output: matched
                points_device,
                points_device + N,
                box_lower_bounds_device,
                min_coord,
                box_width,
                n_boxes,
                n_total_boxes,
                N,
                item);
        }
    );
    // myQueue.wait_and_throw();

    /*
     * Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
     */
    // TODO: We can stream-parallelize these two interpolation functions
    // Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
    num_blocks = (N * n_interpolation_points + num_threads - 1) / num_threads;
    auto e2 = myQueue.parallel_for<class unterpolate_device2>(
        sycl::nd_range<1>(num_blocks * num_threads, num_threads), e1,
        [=](sycl::nd_item<1> item) {

            interpolate_device(
                x_interpolated_values_device,   // output: matched
                x_in_box_device,
                y_tilde_spacings_device,
                denominator_device,
                n_interpolation_points,
                N,
                item);
        }
    );
    // myQueue.wait_and_throw(); // TODO: Remove the synchronization here

    // Compute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
    auto e3 = myQueue.parallel_for<class interpolate_device2>(
        sycl::nd_range<1>(num_blocks * num_threads, num_threads), e1,
        [=](sycl::nd_item<1> item) {

            interpolate_device(
                y_interpolated_values_device,   // output: matched
                y_in_box_device,
                y_tilde_spacings_device,
                denominator_device,
                n_interpolation_points,
                N,
                item);
        }
    );
    // myQueue.wait_and_throw();

    //TODO: Synchronization required here

    // TODO: This section has an atomic-add, can we remove it?
    num_blocks = (n_terms * n_interpolation_points * n_interpolation_points * N + num_threads - 1) / num_threads;
    auto e4 = myQueue.parallel_for<class compute_interpolated_indices2>(
        sycl::nd_range<1>(num_blocks * num_threads, num_threads), {std::move(e2), std::move(e3)},
        [=](sycl::nd_item<1> item) {

            compute_interpolated_indices(
                w_coefficients_device,          // output: matched
                point_box_idx_device,
                chargesQij_device,
                x_interpolated_values_device,
                y_interpolated_values_device,
                N,
                n_interpolation_points,
                n_boxes,
                n_terms,
                item);
        }
    );
    // myQueue.wait_and_throw();

    /*
     * Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
     */
    num_blocks = ((n_terms * n_fft_coeffs_half * n_fft_coeffs_half) + num_threads - 1) / num_threads;
    auto e5 = myQueue.parallel_for<class copy_to_fft_input2>(
        sycl::nd_range<1>(num_blocks * num_threads, num_threads), std::move(e4),
        [=](sycl::nd_item<1> item) {

            copy_to_fft_input(
                fft_input,              // output: matched
                w_coefficients_device,
                n_fft_coeffs,
                n_fft_coeffs_half,
                n_terms,
                item);
        }
    );
    // myQueue.wait_and_throw();

    // Compute fft values at interpolated nodes
    // oneapi::mkl::dft::compute_forward(
    //         *plan_dft,
    //         reinterpret_cast<float *>(fft_input),
    //         (float *)reinterpret_cast<sycl::float2 *>(fft_w_coefficients) ).wait();

    int num_rows = n_fft_coeffs;
    int num_cols = n_fft_coeffs;

    sycl::range<2> block_size(BS1, BS2);
    sycl::range<2> grid_size1((num_rows       + block_size[0] - 1) / block_size[0], (num_cols + block_size[1] - 1) / block_size[1]);
    sycl::range<2> grid_size2(((num_cols/2+1) + block_size[0] - 1) / block_size[0], (num_rows + block_size[1] - 1) / block_size[1]);

    std::vector<sycl::event> events = {sycl::event(), sycl::event(), sycl::event(), sycl::event()};
    for (int f = 0; f < n_terms; ++f) {
        events[f] = myQueue.parallel_for<class DFT2D1gpu2>(
            sycl::nd_range<2>(grid_size1 * block_size, block_size), e5,
            [=](sycl::nd_item<2> item) {

                DFT2D1gpu(
                    fft_input              + f * num_rows * num_cols,
                    fft_scratchpad_device  + f * num_rows * num_cols,
                    num_rows,
                    num_cols,
                    item
                );
            }
        );
        // myQueue.wait_and_throw();

        myQueue.parallel_for<class DFT2D2gpu2>(
            sycl::nd_range<2>(grid_size2 * block_size, block_size), events[f],
            [=](sycl::nd_item<2> item) {

                DFT2D2gpu(
                    fft_scratchpad_device  + f * num_rows * num_cols,
                    fft_w_coefficients     + f * num_rows * (num_cols/2+1),
                    (num_cols/2+1),
                    num_rows,
                    item
                );
            }
        );
    }
    myQueue.wait_and_throw();

    // Take the broadcasted Hadamard product of a complex matrix and a complex vector
    // TODO: Check timing on this kernel
    tsnecuda::utils::BroadcastMatrixVector(
        fft_w_coefficients,                     // 4 x 780 x (780 / 2 + 1) = 4 x 304980 = 1219920 (input/output)
        fft_kernel_tilde_device,                //     780 x  780 = 608400                        (input)
        n_fft_coeffs * (n_fft_coeffs / 2 + 1),  //     780 x (780 / 2 + 1) = 304980
        n_terms,                                // 4
        std::multiplies<std::complex<float>>(),
        0,
        std::complex<float>(1.0f),
        myQueue);

    // Invert the computed values at the interpolated nodes
    // oneapi::mkl::dft::compute_backward(
    //     *plan_idft,
    //     (float *)reinterpret_cast<sycl::float2 *>(fft_w_coefficients),
    //     reinterpret_cast<float *>(fft_output)).wait();

    std::vector<sycl::event> events2 = {sycl::event(), sycl::event(), sycl::event(), sycl::event()};
    for (int f = 0; f < n_terms; ++f) {
        events[f] = myQueue.parallel_for<class iDFT2D1gpu2>(
            sycl::nd_range<2>(grid_size1 * block_size, block_size),
            [=](sycl::nd_item<2> item) {

                iDFT2D1gpu(
                    fft_w_coefficients     + f * num_rows * (num_cols/2+1),
                    fft_scratchpad_device  + f * num_rows * num_cols,
                    num_rows,
                    num_cols,
                    item
                );
            }
        );
        // myQueue.wait_and_throw();

        events2[f] = myQueue.parallel_for<class iDFT2D2gpu2>(
            sycl::nd_range<2>(grid_size1 * block_size, block_size), events[f],
            [=](sycl::nd_item<2> item) {

                iDFT2D2gpu(
                    fft_scratchpad_device  + f * num_rows * num_cols,
                    fft_output             + f * num_rows * num_cols,
                    num_cols,
                    num_rows,
                    item
                );
            }
        );
    }
    // myQueue.wait_and_throw();

    auto e6 = myQueue.parallel_for<class copy_from_fft_output2>(
        sycl::nd_range<1>(num_blocks * num_threads, num_threads), events2,
        [=](sycl::nd_item<1> item) {

            copy_from_fft_output(
                y_tilde_values,         // output
                fft_output,
                n_fft_coeffs,
                n_fft_coeffs_half,
                n_terms,
                item);
        }
    );
    // myQueue.wait_and_throw();

    /*
     * Step 3: Compute the potentials \tilde{\phi}
     */
    // TODO: Depending on the profiling here, we should check to see if we can split this code
    num_blocks = (n_terms * n_interpolation_points * n_interpolation_points * N + num_threads - 1) / num_threads;
    myQueue.parallel_for<class compute_potential_indices2>(
        sycl::nd_range<1>(num_blocks * num_threads, num_threads), std::move(e6),
        [=](sycl::nd_item<1> item) {

            compute_potential_indices(
                potentialsQij_device,           // output
                point_box_idx_device,
                y_tilde_values,
                x_interpolated_values_device,
                y_interpolated_values_device,
                N,
                n_interpolation_points,
                n_boxes,
                n_terms,
                item);
        }
    );
    myQueue.wait_and_throw();
}
