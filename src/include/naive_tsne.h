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
     * @brief Compute the Pij distribution O(n^2)
     * 
     * @param handle CUBLAS handle
     * @param points The points in an NxNDIM column-major array
     * @param sigma The list of sigmas for each point
     * @param pij The computed pij value
     * @param N The number of points
     * @param NDIMS The number of dimensions for each point
     */
    void compute_pij(cublasHandle_t &handle, 
                thrust::device_vector<float> &pij,  
                const thrust::device_vector<float> &points, 
                const thrust::device_vector<float> &sigma,
                const unsigned int N, 
                const unsigned int NDIMS);
    /**
     * @brief Compute the Pij based on P(i|j) using Pij = P(i|j) + P(j|i)/2N O(n^2)
     * 
     * @param handle CUBLAS handle
     * @param pij_vals Computed P(i|j) values
     * @param N The number of points
     */
    void symmetrize_pij(cublasHandle_t &handle, 
                            thrust::device_vector<float> &pij, 
                            const unsigned int N);

    /**
     * @brief Searches the right sigmas for computing pij
     * 
     * @param handle CUBLAS handle
     * @param points Original Points
     * @param perplexity_target Target perplexity for the Search
     * @param eps Error tolerance for the perplexity search
     * @param N The number of points
     * @param NDIMS Number of Dimensions of points
     * @return thrust::device_vector<float> Computed Sigmas based on the perplexity target
     */
    thrust::device_vector<float> search_perplexity(cublasHandle_t &handle,
                        thrust::device_vector<float> &points,
                        const float perplexity_target,
                        const float eps,
                        const unsigned int N,
                        const unsigned int NDIMS);

    ///@private
    void thrust_search_perplexity(cublasHandle_t &handle,
                        thrust::device_vector<float> &sigmas,
                        thrust::device_vector<float> &lower_bound,
                        thrust::device_vector<float> &upper_bound,
                        thrust::device_vector<float> &perplexity,
                        const thrust::device_vector<float> &pij,
                        const float target_perplexity,
                        const unsigned int N);

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
