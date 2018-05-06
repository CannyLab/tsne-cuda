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

namespace tsnecuda {
namespace util {

struct FunctionalExp {
    __host__ __device__ float operator()(const float &x) const {
        return exp(x);
    }
};
struct FunctionalIncrementInverse {
    __host__ __device__ float operator()(const float &x) const {
        return 1 / (x + 1);
    }
};
struct FunctionalKlDivergence {
    __host__ __device__ float operator()(const float &x,
            const float &y) const {
        return x == 0.0f ? 0.0f : x * (log(x) - log(y));
    }
};
struct FunctionalEntropy {
  __host__ __device__ float operator()(const float &x) const {
      float val = x*log(x);
      return (val != val || isinf(val)) ? 0 : val;
    }
};
struct FunctionalPower2 {
    __host__ __device__ float operator()(const float &x) const {
        return pow(2, x);
    }
};
struct FunctionalSqrt {
    __host__ __device__ float operator()(const float &x) const {
        return pow(x, 0.5);
    }
};
struct FunctionalSquare {
    __host__ __device__ float operator()(const float &x) const {
        return x * x;
    }
};
struct FunctionalAbs {
    __host__ __device__ float operator()(const float &x) const {
        return fabsf(x);
    }
};
struct FunctionalNanOrInf {
    __host__ __device__ bool operator()(const float &x) const {
        return isnan(x) || isinf(x);
    }
};

}  // namespace util
}  // namespace tsnecuda

#endif  // SRC_INCLUDE_UTIL_THRUST_TRANSFORM_FUNCTIONS_H_
