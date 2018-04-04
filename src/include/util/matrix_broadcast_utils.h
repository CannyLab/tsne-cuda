/**
 * @brief Utilities for broadcasting across a GPU matrix
 * 
 * @file matrix_broadcast.h
 * @author David Chan
 * @date 2018-04-04
 */

#ifndef MATRIX_BROADCAST_H
#define MATRIX_BROADCAST_H

    #include "common.h"
    #include "util/cuda_utils.h"

    template<typename BinaryFunction, typename T>
    __global__ void _broadcast_row_vec(T * __restrict__ matrix, 
                            const T * __restrict__ vector, 
                            const unsigned int N, 
                            const unsigned int M, 
                            BinaryFunction binary_op, 
                            const T alpha);
    
    template<typename BinaryFunction, typename T>
    __global__ void _broadcast_col_vec(T * __restrict__ matrix, 
                            const T * __restrict__ vector, 
                            const unsigned int N, 
                            const unsigned int M,
                            BinaryFunction binary_op,
                            const T alpha);

    template<typename BinaryFunction, typename T>
    void broadcast_matrix_vector(thrust::device_vector<T> &matrix, 
                                const thrust::device_vector<T> &vector, 
                                const unsigned int N, 
                                const unsigned int M, 
                                BinaryFunction binary_op,
                                const unsigned int axis,
                                const T alpha);
#endif
