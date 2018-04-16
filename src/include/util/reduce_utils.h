/**
 * @brief Utilities for reduction across a matrix on an axis
 * 
 * @file reduce_utils.h
 * @author David Chan
 * @date 2018-04-04
 */

#ifndef REDUCE_UTILS_H
#define REDUCE_UTILS_H

    #include "common.h"
    #include "util/cuda_utils.h"

    namespace Reduce {

        /**
         * @brief Reduce a matrix by summing then multiplying by alpha along the reduction axis
         * 
         * @param handle CUBLAS handle
         * @param matrix The NxM matrix to reduce
         * @param N The number of rows in the matrix
         * @param M The number of columns in the matrix
         * @param alpha The alpha to multiply by
         * @param axis The axis to reduce on (0 = rows, 1 = cols)
         * @return thrust::device_vector<float> The reduced vector 
         */
        thrust::device_vector<float> reduce_alpha(cublasHandle_t &handle, 
                                        const thrust::device_vector<float> &matrix, 
                                        const unsigned int N, 
                                        const unsigned int M, 
                                        float alpha, 
                                        const int axis);

        /**
         * @brief Reduce a matrix by computing the mean of the reduction axis
         * 
         * @param handle CUBLAS handle
         * @param matrix The NxM matrix to reduce
         * @param N The number of rows in the matrix
         * @param M The number of columns in the matrix
         * @param axis The axis to reduce on (0 = rows, 1 = cols)
         * @return thrust::device_vector<float> The reduced vector 
         */
        thrust::device_vector<float> reduce_mean(cublasHandle_t &handle, 
                                            const thrust::device_vector<float> &matrix, 
                                            const unsigned int N, 
                                            const unsigned int M, 
                                            const int axis);

        /**
         * @brief Reduce a matrix by computing the sum of the reduction axis
         * 
         * @param handle CUBLAS handle
         * @param matrix The NxM matrix to reduce
         * @param N The number of rows in the matrix
         * @param M The number of columns in the matrix
         * @param axis The axis to reduce on (0 = rows, 1 = cols)
         * @return thrust::device_vector<float> The reduced vector 
         */
        thrust::device_vector<float> reduce_sum(cublasHandle_t &handle, 
                                            const thrust::device_vector<float> &matrix, 
                                            const unsigned int N, 
                                            const unsigned int M, 
                                            const int axis); 

    }

                                       

#endif