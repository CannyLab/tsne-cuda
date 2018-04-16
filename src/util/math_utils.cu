/**
 * @brief 
 * 
 * @file math_utils.cu
 * @author David Chan
 * @date 2018-04-04
 */

 #include "util/math_utils.h"

 struct func_square {
    __host__ __device__ float operator()(const float &x) const { return x * x; }
};
struct func_sqrt {
    __host__ __device__ float operator()(const float &x) const { return pow(x, 0.5); }
};
struct func_abs {
    __host__ __device__ float operator()(const float &x) const { return fabs(x); }
};
struct func_nan_or_inf {
    __host__ __device__ bool operator()(const float &x) const { return isnan(x) || isinf(x); }
};

void Math::gauss_normalize(cublasHandle_t &handle, thrust::device_vector<float> &points, const unsigned int N, const unsigned int NDIMS) {
    auto means = Reduce::reduce_mean(handle, points, N, NDIMS, 0);

    // zero center
    Broadcast::broadcast_matrix_vector(points, means, N, NDIMS, thrust::minus<float>(), 1, 1.f);
    
    // compute standard deviation
    thrust::device_vector<float> squared_vals(points.size());
    Math::square(points, squared_vals);
    auto norm_sum_of_squares = Reduce::reduce_alpha(handle, squared_vals, N, NDIMS, 1.f / (N - 1), 0);
    thrust::device_vector<float> stddev(norm_sum_of_squares.size());
    Math::sqrt(norm_sum_of_squares, stddev);

    // normalize
    Broadcast::broadcast_matrix_vector(points, stddev, N, NDIMS, thrust::divides<float>(), 1, 1.f);
}

void Math::square(const thrust::device_vector<float> &vec, thrust::device_vector<float> &out) {
    thrust::transform(vec.begin(), vec.end(), out.begin(), func_square());
}

void Math::sqrt(const thrust::device_vector<float> &vec, thrust::device_vector<float> &out) {
    thrust::transform(vec.begin(), vec.end(), out.begin(), func_sqrt());
}

float Math::norm(const thrust::device_vector<float> &vec) {
    return std::sqrt( thrust::transform_reduce(vec.begin(), vec.end(), func_square(), 0.0f, thrust::plus<float>()) );
}

bool Math::any_nan_or_inf(const thrust::device_vector<float> &vec) {
    return thrust::transform_reduce(vec.begin(), vec.end(), func_nan_or_inf(), 0, thrust::plus<bool>());
}

void Math::max_norm(thrust::device_vector<float> &vec) {
    float max_val = thrust::transform_reduce(vec.begin(), vec.end(), func_abs(), 0.0f, thrust::maximum<float>());
    thrust::constant_iterator<float> div_iter(max_val);
    thrust::transform(vec.begin(), vec.end(), div_iter, vec.begin(), thrust::divides<float>());
}
