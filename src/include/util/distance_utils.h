/**
 * @brief Utilities for the computation of various distances
 * 
 * @file distance_utils.h
 * @author David Chan
 * @date 2018-04-04
 */

#ifndef DISTANCE_UTILS_H
#define DISTANCE_UTILS_H

    #include "common.h"
    #include "util/cuda_utils.h"
    #include "util/reduce_utils.h"
    #include "util/math_utils.h"
    #include "util/thrust_utils.h"

    __global__ void assemble_final_result(const float * __restrict__ d_norms_x_2, 
                                          float * __restrict__ d_dots,
                                          const int N);
    void squared_pairwise_dist(cublasHandle_t &handle, 
                    thrust::device_vector<float> &distances, 
                    const thrust::device_vector<float> &points, 
                    const unsigned int N, 
                    const unsigned int NDIMS);

    void pairwise_dist(cublasHandle_t &handle, 
                    thrust::device_vector<float> &distances, 
                    const thrust::device_vector<float> &points, 
                    const unsigned int N, 
                    const unsigned int NDIMS);
#endif
