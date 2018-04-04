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

    thrust::device_vector<float> reduce_alpha(cublasHandle_t &handle, 
                                          const thrust::device_vector<float> &matrix, 
                                          const unsigned int N, 
                                          const unsigned int M, 
                                          float alpha, 
                                          const int axis);
    thrust::device_vector<float> reduce_mean(cublasHandle_t &handle, 
                                         const thrust::device_vector<float> &matrix, 
                                         const unsigned int N, 
                                         const unsigned int M, 
                                         const int axis);
    thrust::device_vector<float> reduce_sum(cublasHandle_t &handle, 
                                        const thrust::device_vector<float> &matrix, 
                                        const unsigned int N, 
                                        const unsigned int M, 
                                        const int axis);                                     

#endif