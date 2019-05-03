#ifndef NBODYFFT_H
#define NBODYFFT_H

#include <complex>
#include "common.h"
#include <thrust/complex.h>

using namespace std;

typedef float (*kernel_type)(float, float);

typedef float (*kernel_type_2d)(float, float, float, float);

void precompute_2d(cufftHandle &plan_kernel_tilde, float x_max, float x_min, float y_max, float y_min, int n_boxes, int n_interpolation_points,
                   thrust::device_vector<float> &box_lower_bounds_device, thrust::device_vector<float> &box_upper_bounds_device, 
                   thrust::device_vector<float> &kernel_tilde_device, thrust::device_vector<thrust::complex<float>> &fft_kernel_tilde_device);

void n_body_fft_2d(
    cufftHandle &plan_dft,
    cufftHandle &plan_idft,
    int N, 
    int n_terms, 
    int n_boxes,
    int n_interpolation_points, 
    thrust::device_vector<thrust::complex<float>> &fft_kernel_tilde_device,
    int n_total_boxes,
    int total_interpolation_points,
    float coord_min,
    float box_width,
    int n_fft_coeffs_half,
    int n_fft_coeffs,
    int num_nodes,
    thrust::device_vector<float> &fft_input,
    thrust::device_vector<thrust::complex<float>> &fft_w_coefficients,
    thrust::device_vector<float> &fft_output,
    thrust::device_vector<int> &point_box_idx_device, 
    thrust::device_vector<float> &x_in_box_device,
    thrust::device_vector<float> &y_in_box_device,
    thrust::device_vector<float> &points_device,
    thrust::device_vector<float> &box_lower_bounds_device,
    thrust::device_vector<float> &y_tilde_spacings_device,
    thrust::device_vector<float> &denominator_device,
    thrust::device_vector<float> &y_tilde_values,
    thrust::device_vector<float> &all_interpolated_values_device,
    thrust::device_vector<float> &output_values,
    thrust::device_vector<int> &all_interpolated_indices,
    thrust::device_vector<int> &output_indices,
    thrust::device_vector<float> &w_coefficients_device,
    thrust::device_vector<float> &chargesQij_device,
    thrust::device_vector<float> &x_interpolated_values_device,
    thrust::device_vector<float> &y_interpolated_values_device,
    thrust::device_vector<float> &potentialsQij_device);

void interpolate(int n_interpolation_points, int N, const float *y_in_box, const float *y_tilde_spacings,
                 float *interpolated_values, const float *denominator);

float* get_ntime();

#endif
