/**
 * @brief Utilities for broadcasting across a GPU matrix
 * 
 * @file matrix_broadcast.h
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_MATRIX_BROADCAST_UTILS_H_
#define SRC_INCLUDE_UTIL_MATRIX_BROADCAST_UTILS_H_

#include "include/common.h"
#include "include/util/cuda_utils.h"

namespace tsnecuda {
namespace util {

/// @private
template<typename BinaryFunction, typename T>
__global__ void BroadcastRowVector(T * __restrict__ matrix,
                        const T * __restrict__ vector,
                        const int N,
                        const int M,
                        BinaryFunction binary_operation,
                        const T alpha);

/// @private
template<typename BinaryFunction, typename T>
__global__ void BroadcastColumnVector(T * __restrict__ matrix,
                        const T * __restrict__ vector,
                        const int N,
                        const int M,
                        BinaryFunction binary_operation,
                        const T alpha);


/**
* @brief Function for broadcasting a vector across a matrix
* 
* @tparam BinaryFunction The function to broadcast with
* @tparam T Matrix format
* @param matrix (N x M) matrix stored in column major order
* @param vector Length N vector if axis == 0, length M vector if axis == 1
* @param N,M dimensions of matrix
* @param binary_operation an operation that takes in two arguments of type T and returns a type T
* @param axis 0 or 1, controlls whether this runs a column or row broadcast
* @param alpha scalar multiple for vector
* 
* @note 
* should axis == 0 be row or column broadcasting? and vice versa for axis == 1?
*/
template<typename BinaryFunction, typename T>
void BroadcastMatrixVector(thrust::device_vector<T> &d_matrix,
                            const thrust::device_vector<T> &d_vector,
                            const int N,
                            const int M,
                            BinaryFunction binary_operation,
                            const int axis,
                            const T alpha);
}  // namespace util
}  // namespace tsnecuda

#endif  // SRC_INCLUDE_UTIL_MATRIX_BROADCAST_UTILS_H_
