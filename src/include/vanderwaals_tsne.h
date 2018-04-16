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

namespace VWS {
    /**
     * @brief Project points into the lower-dimensional space by doing a simple cutoff-based
     * particle simulation. 
     * 
     * @param handle CUBLAS handle
     * @param points The points to project, Column-Major (NxNDIM)
     * @param N The number of points
     * @param NDIMS The number of dimensions in the original space
     * @param PROJDIM The number of dimensions to project to
     * @return thrust::device_vector<float> The projected points
     */
    thrust::device_vector<float> tsne(cublasHandle_t &handle, 
                                    thrust::device_vector<float> &points, 
                                    const unsigned int N, 
                                    const unsigned int NDIMS,
                                    const unsigned int PROJDIM);



}

#endif