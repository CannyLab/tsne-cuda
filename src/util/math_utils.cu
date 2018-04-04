/**
 * @brief 
 * 
 * @file math_utils.cu
 * @author David Chan
 * @date 2018-04-04
 */

 #include "util/math_utils.h"

 struct func_square {
    __host__ __device__ double operator()(const float &x) const { return x * x; }
};
struct func_sqrt {
    __host__ __device__ double operator()(const float &x) const { return pow(x, 0.5); }
};
 struct func_abs {
    __host__ __device__ double operator()(const float &x) const { return fabs(x); }
};

 void gauss_normalize(cublasHandle_t &handle, thrust::device_vector<float> &points, const unsigned int N, const unsigned int NDIMS) {
    auto means = reduce_mean(handle, points, N, NDIMS, 0);

    // zero center
    broadcast_matrix_vector(points, means, N, NDIMS, thrust::minus<float>(), 1, 1.f);
    
    // compute standard deviation
    auto squared_vals = square(points, N * NDIMS);
    auto norm_sum_of_squares = reduce_alpha(handle, squared_vals, N, NDIMS, 1.f / (N - 1), 0);
    auto stddev = sqrt(norm_sum_of_squares, N * NDIMS);

    // normalize
    broadcast_matrix_vector(points, stddev, N, NDIMS, thrust::divides<float>(), 1, 1.f);
}

thrust::device_vector<float> square(const thrust::device_vector<float> &vec, const unsigned int N) {
    thrust::device_vector<float> squared_vals(N);
    thrust::transform(vec.begin(), vec.end(), squared_vals.begin(), func_square());
    return squared_vals;
}

thrust::device_vector<float> sqrt(const thrust::device_vector<float> &vec, const unsigned int N) {
    thrust::device_vector<float> sqrt_vals(N);
    thrust::transform(vec.begin(), vec.end(), sqrt_vals.begin(), func_sqrt());
    return sqrt_vals;
}

void max_norm(thrust::device_vector<float> &vec) {
    float max_val = thrust::transform_reduce(vec.begin(), vec.end(), func_abs(), 0.0f, thrust::maximum<float>());
    thrust::constant_iterator<float> div_iter(max_val);
    thrust::transform(vec.begin(), vec.end(), div_iter, vec.begin(), thrust::divides<float>());
}