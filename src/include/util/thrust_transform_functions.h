/**
 * @brief Various transformations for thrust::transform
 * 
 * @file thrust_transform_functions.h
 * @author David Chan
 * @date 2018-05-05
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_THRUST_TRANSFORM_FUNCTIONS_H_
#define SRC_INCLUDE_UTIL_THRUST_TRANSFORM_FUNCTIONS_H_

#include "include/common.h"

struct func_exp {
    __host__ __device__ float operator()(const float &x) const { return exp(x); }
};
struct func_inc_inv {
    __host__ __device__ float operator()(const float &x) const { return 1 / (x + 1); }
};
struct func_kl {
    __host__ __device__ float operator()(const float &x, const float &y) const { 
        return x == 0.0f ? 0.0f : x * (log(x) - log(y));
    }
};
struct func_entropy_kernel {
  __host__ __device__ float operator()(const float &x) const { float val = x*log(x); return (val != val || isinf(val)) ? 0 : val; }
};
struct func_pow2 {
    __host__ __device__ float operator()(const float &x) const { return pow(2,x); }
};
struct func_sqrt {
    __host__ __device__ float operator()(const float &x) const {
            return pow(x, 0.5); }
};
struct func_square {
    __host__ __device__ float operator()(const float &x) const { return x * x; }
};
struct func_abs {
    __host__ __device__ float operator()(const float &x) const {
        return fabsf(x);
    }
};
struct func_nan_or_inf {
    __host__ __device__ bool operator()(const float &x) const {
        return isnan(x) || isinf(x);
    }
};

#endif  // SRC_INCLUDE_UTIL_THRUST_TRANSFORM_FUNCTIONS_H_
