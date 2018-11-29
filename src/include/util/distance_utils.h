/**
 * @brief Utilities for the computation of various distances
 * 
 * @file distance_utils.h
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_DISTANCE_UTILS_H_
#define SRC_INCLUDE_UTIL_DISTANCE_UTILS_H_

// FAISS includes
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/IndexProxy.h>
#include <faiss/gpu/StandardGpuResources.h>

// CXX Includes
#include <stdint.h>

// Local Includes
#include "include/common.h"
#include "include/options.h"
#include "include/util/cuda_utils.h"
#include "include/util/reduce_utils.h"
#include "include/util/math_utils.h"
#include "include/util/thrust_utils.h"

namespace tsnecuda {
namespace util {

/**
 * @brief Assemble the final distances from the squared norms and the dot produces
 * 
 * @param d_squared_norms The squared norms of the points that you want to handle (Device Float)
 * @param d_dot_products The dot products (Device Float)
 * @param num_points The number of points in the arrays
 */
__global__ void AssembleDistances(
        const float * __restrict__ d_squared_norms,
        float * __restrict__ d_dot_products,
        const int num_points);

/**
* @brief Compute the squared pairwise euclidean distance between the given points
* 
* @param handle CUBLAS handle reference
* @param d_distances The output device vector (must be NxN)
* @param d_points Vector of points in Column-major format (NxNDIM matrix)
* @param num_points The number of points in the vectors
* @param num_dims The number of dimensions per point
*/
void SquaredPairwiseDistance(cublasHandle_t &handle,
        thrust::device_vector<float> &d_distances,
        const thrust::device_vector<float> &d_points,
        const int num_points,
        const int num_dims);

/**
* @brief Compute the pairwise euclidean distance between the given poitns
* 
* @param handle CUBLAS handle reference
* @param d_distances The output device vector (must be NxN)
* @param d_points Vector of points in Column-major format (NxNDIM matrix)
* @param num_points The number of points in the vectors
* @param num_dims The number of dimensions per point
*/
void PairwiseDistance(cublasHandle_t &handle, 
        thrust::device_vector<float> &d_distances,
        const thrust::device_vector<float> &d_points,
        const int num_points,
        const int num_dims);


/**
* @brief Use FAISS to compute the k-nearest neighbors for the given points
* 
* @param gpu_opt GPU Options object
* @param indices The index array that goes with the distance array (N_POINTSxK) row-major (so I[K*i + j] gives the j'th nearest neighbor of the i'th point)
* @param distances The euclidean distance array (true euclidean distance, not squared) (N_POINTSxK) row-major
* @param points The points of which you want the k nearest neighbors (N_POINTSxN_DIMS) row-major (so points[N_DIM*i + j] gives the j'th dim of the i'th point)
* @param num_dims The number of dimensions of the input points
* @param num_points The number of input points
* @param K The number of nearest neighbors to return. If >=1024, this function uses the CPU instead of the GPU
*/
void KNearestNeighbors(tsnecuda::GpuOptions& gpu_opt, int64_t* indices, float* distances,
        const float* const points,
        const int num_dims, const int num_points,
        const int num_near_neighbots);

__global__
void PostprocessNeighborIndicesKernel(
                                    volatile int * __restrict__ indices,
                                    const long * __restrict__ long_indices,
                                    const int num_points,
                                    const int num_neighbors); 

void PostprocessNeighborIndices(
                tsnecuda::GpuOptions &gpu_opt,
                thrust::device_vector<int> &indices,
                thrust::device_vector<int64_t> &long_indices,
                const int num_points,
                const int num_neighbors);

}
}
#endif  // SRC_INCLUDE_UTIL_DISTANCE_UTILS_H_
