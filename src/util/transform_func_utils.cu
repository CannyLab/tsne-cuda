/**
 * @brief TODO: Write this
 * 
 * @file transform_func_utils.cu
 * @author David Chan
 * @date 2018-04-04
 */


#include "util/transform_func_utils.h"

__host__ __device__ double tfunc::square::operator()(const float &x) const { return x * x; }
__host__ __device__ double tfunc::sqrt::operator()(const float &x) const { return pow(x, 0.5); }
__host__ __device__ double tfunc::exp::operator()(const float &x) const { return exp(x); }
__host__ __device__ double tfunc::exp_no_zero::operator()(const float &x) const { return x < -1e-4 ? exp(x) : 0; }
__host__ __device__ double tfunc::inv::operator()(const float &x) const { return pow(x, -1.0); }
__host__ __device__ double tfunc::inc::operator()(const float &x) const { return x + 1; }
__host__ __device__ double tfunc::inc_inv::operator()(const float &x) const { return pow(x + 1, -1.0); }
__host__ __device__ double tfunc::inc_inv_ignore_zero::operator()(const float &x) const { return x > 1e-4 ? pow(x + 1, -1.0) : 0; }
__host__ __device__ double tfunc::ln::operator()(const float &x) const { return log(x); }
__host__ __device__ double tfunc::kl::operator()(const float &x, const float &y) const { return y < 1e-4 ? 0 : x * log(x / y); }
__host__ __device__ double tfunc::abs::operator()(const float &x) const { return fabs(x); }