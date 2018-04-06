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
    void square(const thrust::device_vector<float> &vec, thrust::device_vector<float> &out);
    void sqrt(const thrust::device_vector<float> &vec, thrust::device_vector<float> &out);
    float norm(const thrust::device_vector<float> &vec);
    bool any_nan_or_inf(const thrust::device_vector<float> &vec);
    void max_norm(thrust::device_vector<float> &vec);

#endif
