/**
 * @brief Short-range forces implementation of T-SNE O(n)
 * 
 * @file vanderwaals_tsne.h
 * @author David Chan
 * @date 2018-04-16
 */

#ifndef VWS_TSNE_H
#define VWS_TSNE_H

#include "common.h"
#include "util/cuda_utils.h"
#include "util/math_utils.h"
#include "util/matrix_broadcast_utils.h"
#include "util/reduce_utils.h"
#include "util/distance_utils.h"
#include "util/random_utils.h"
#include "util/thrust_utils.h"
#include "naive_tsne.h"

namespace vws {
    thrust::device_vector<float> compute_pij(cublasHandle_t &handle, 
                                            thrust::device_vector<float> &points, 
                                            thrust::device_vector<float> &sigma, 
                                            const unsigned int N, 
                                            const unsigned int NDIMS);
    float compute_gradients(cublasHandle_t &handle, 
                            thrust::device_vector<float> &forces,
                            thrust::device_vector<float> &dist, 
                            thrust::device_vector<float> &ys, 
                            thrust::device_vector<float> &pij, 
                            thrust::device_vector<float> &qij, 
                            const unsigned int N,
                            float eta);
    thrust::device_vector<float> naive_tsne(cublasHandle_t &handle, 
                                        thrust::device_vector<float> &points, 
                                        const unsigned int N, 
                                        const unsigned int NDIMS,
                                        const unsigned int PROJDIM);
}

#endif