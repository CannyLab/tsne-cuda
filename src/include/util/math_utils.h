/**
 * @brief Utilities for doing math of different sorts
 * 
 * @file math_utils.h
 * @author David Chan
 * @date 2018-04-04
 */


#ifndef MATH_UTILS_H
#define MATH_UTILS_H

    #include "common.h"
    #include "util/matrix_broadcast_utils.h"
    #include "util/reduce_utils.h"

    void gauss_normalize(cublasHandle_t &handle, thrust::device_vector<float> &points, const unsigned int N, const unsigned int NDIMS);
    thrust::device_vector<float> square(const thrust::device_vector<float> &vec, const unsigned int N);
    thrust::device_vector<float> sqrt(const thrust::device_vector<float> &vec, const unsigned int N);
    void max_norm(thrust::device_vector<float> &vec);

#endif