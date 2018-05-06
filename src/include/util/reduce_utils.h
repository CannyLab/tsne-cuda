/**
 * @brief Utilities for reduction across a matrix on an axis
 * 
 * @file reduce_utils.h
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_REDUCE_UTILS_H_
#define SRC_INCLUDE_UTIL_REDUCE_UTILS_H_

#include "include/common.h"
#include "include/util/cuda_utils.h"

namespace tsne {
namespace util {

/**
* @brief Reduce a matrix by summing then multiplying by alpha along the reduction axis
* 
* @param handle CUBLAS handle
* @param d_matrix The NxM matrix to reduce
* @param N The number of rows in the matrix
* @param M The number of columns in the matrix
* @param alpha The alpha to multiply by
* @param axis The axis to reduce on (0 = rows, 1 = cols)
* @return thrust::device_vector<float> The reduced vector 
*/
thrust::device_vector<float> ReduceAlpha(cublasHandle_t &handle,
                            const thrust::device_vector<float> &d_matrix,
                            const uint32_t N,
                            const uint32_t M,
                            float alpha,
                            const uint32_t axis);

/**
* @brief Reduce a matrix by computing the mean of the reduction axis
* 
* @param handle CUBLAS handle
* @param d_matrix The NxM matrix to reduce
* @param N The number of rows in the matrix
* @param M The number of columns in the matrix
* @param axis The axis to reduce on (0 = rows, 1 = cols)
* @return thrust::device_vector<float> The reduced vector 
*/
thrust::device_vector<float> ReduceMean(cublasHandle_t &handle,
                                const thrust::device_vector<float> &d_matrix,
                                const uint32_t N,
                                const uint32_t M,
                                const uint32_t axis);

/**
* @brief Reduce a matrix by computing the sum of the reduction axis
* 
* @param handle CUBLAS handle
* @param d_matrix The NxM matrix to reduce
* @param N The number of rows in the matrix
* @param M The number of columns in the matrix
* @param axis The axis to reduce on (0 = rows, 1 = cols)
* @return thrust::device_vector<float> The reduced vector 
*/
thrust::device_vector<float> ReduceSum(cublasHandle_t &handle,
                                const thrust::device_vector<float> &d_matrix,
                                const uint32_t N,
                                const uint32_t M,
                                const uint32_t axis);

}  // namespace util
}  // namespace tsne

#endif  // SRC_INCLUDE_UTIL_REDUCE_UTILS_H_
