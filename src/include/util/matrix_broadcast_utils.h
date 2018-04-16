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

    namespace Broadcast {
        
        /// @private
        template<typename BinaryFunction, typename T>
        __global__ void _broadcast_row_vec(T * __restrict__ matrix, 
                                const T * __restrict__ vector, 
                                const unsigned int N, 
                                const unsigned int M, 
                                BinaryFunction binary_op, 
                                const T alpha);
        /// @private
        template<typename BinaryFunction, typename T>
        __global__ void _broadcast_col_vec(T * __restrict__ matrix, 
                                const T * __restrict__ vector, 
                                const unsigned int N, 
                                const unsigned int M,
                                BinaryFunction binary_op,
                                const T alpha);


         /**
        * @brief Function for broadcasting a vector across a matrix
        * 
        * @tparam BinaryFunction The function to broadcast with
        * @tparam T Matrix format
        * @param matrix (N x M) matrix stored in column major order
        * @param vector Length N vector if axis == 0, length M vector if axis == 1
        * @param N,M dimensions of matrix
        * @param binary_op an operation that takes in two arguments of type T and returns a type T
        * @param axis 0 or 1, controlls whether this runs a column or row broadcast
        * @param alpha scalar multiple for vector
        * 
        * @note 
        * should axis == 0 be row or column broadcasting? and vice versa for axis == 1?
        *
        *
        */
        template<typename BinaryFunction, typename T>
        void broadcast_matrix_vector(thrust::device_vector<T> &matrix, 
                                    const thrust::device_vector<T> &vector, 
                                    const unsigned int N, 
                                    const unsigned int M, 
                                    BinaryFunction binary_op,
                                    const unsigned int axis,
                                    const T alpha);
    }

    
#endif
