/**
 * @brief Naive implementation of T-SNE O(n^2)
 * 
 * @file naive_tsne.h
 * @author David Chan
 * @date 2018-04-04
 */

#ifndef NAIVE_TSNE_H
#define NAIVE_TSNE_H

#include "common.h"
#include "util/cuda_utils.h"
#include "util/math_utils.h"
#include "util/matrix_broadcast_utils.h"
#include "util/reduce_utils.h"
#include "util/distance_utils.h"
#include "util/random_utils.h"
#include "util/thrust_utils.h"

namespace NaiveTSNE {
    /**
     * @brief Compute the P(i|j) distribution O(n^2)
     * 
     * @param handle CUBLAS handle
     * @param points The points in an NxNDIM column-major array
     * @param sigma The list of sigmas for each point
     * @param N The number of points
     * @param NDIMS The number of dimensions for each point
     * @return thrust::device_vector<float> The computer P(i|j) distribution (symmetrized)
     */
    thrust::device_vector<float> compute_pij(cublasHandle_t &handle, 
                                         thrust::device_vector<float> &points, 
                                         thrust::device_vector<float> &sigma, 
                                         const unsigned int N, 
                                         const unsigned int NDIMS);

    /**
     * @brief Compute the T-SNE gradients 
     * 
     * @param handle CUBLAS handle
     * @param forces The forces output array
     * @param dist Placeholder array for the distances
     * @param ys The current projected points
     * @param pij The P(i|j) distribution
     * @param qij Placeholder for the computed Q(i|j) distribution
     * @param N The number of points
     * @param PROJDIM The number of dimensions to project into
     * @param eta The learning rate
     * @return float Returns the K-L Divergence of the two distributions (as the loss)
     */
    float compute_gradients(cublasHandle_t &handle, 
                            thrust::device_vector<float> &forces,
                            thrust::device_vector<float> &dist, 
                            thrust::device_vector<float> &ys, 
                            thrust::device_vector<float> &pij, 
                            thrust::device_vector<float> &qij, 
                            const unsigned int N,
                            const unsigned int PROJDIM,
                            float eta);

    /**
     * @brief Perform T-SNE using the naive O(n^2) forces computation
     * 
     * @param handle The CUBLAS handle to use
     * @param points The array of points in column-major NxNDIM matrix
     * @param N The number of points
     * @param NDIMS The number of dimentions the original points are in
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