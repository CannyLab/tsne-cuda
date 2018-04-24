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

    // FAISS includes
    #include <faiss/IndexFlat.h>
    #include <faiss/IndexIVFFlat.h>
    #include <faiss/IndexIVFPQ.h>
    #include <faiss/gpu/GpuIndexFlat.h>
    #include <faiss/gpu/GpuIndexIVFFlat.h>
    #include <faiss/gpu/StandardGpuResources.h>

    namespace Distance {
        /**
         * @brief Kernel for assembling distance results from the norms and dot product vectors
         * 
         * @param d_norms_x_2 The squared norms of the components
         * @param d_dots The dot products of the components (returns the output in here)
         * @param N The length of the vector
         */
         __global__ void assemble_final_result(const float * __restrict__ d_norms_x_2, 
                                          float * __restrict__ d_dots,
                                          const int N);

        /**
         * @brief Compute the squared pairwise euclidean distance between the given points
         * 
         * @param handle CUBLAS handle reference
         * @param distances The output device vector (must be NxN)
         * @param points Vector of points in Column-major format (NxNDIM matrix)
         * @param N The number of points in the vectors
         * @param NDIMS The number of dimensions per point
         */
        void squared_pairwise_dist(cublasHandle_t &handle, 
                        thrust::device_vector<float> &distances, 
                        const thrust::device_vector<float> &points, 
                        const unsigned int N, 
                        const unsigned int NDIMS);

        /**
         * @brief Compute the pairwise euclidean distance between the given poitns
         * 
         * @param handle CUBLAS handle reference
         * @param distances The output device vector (must be NxN)
         * @param points Vector of points in Column-major format (NxNDIM matrix)
         * @param N The number of points in the vectors
         * @param NDIMS The number of dimensions per point
         */
        void pairwise_dist(cublasHandle_t &handle, 
                        thrust::device_vector<float> &distances, 
                        const thrust::device_vector<float> &points, 
                        const unsigned int N, 
                        const unsigned int NDIMS);


        /**
         * @brief Use FAISS to compute the k-nearest neighbors for the given points
         * 
         * @param points The points of which you want the k nearest neighbors (N_POINTSxN_DIMS) row-major (so points[N_DIM*i + j] gives the j'th dim of the i'th point)
         * @param I The index array that goes with the distance array (N_POINTSxK) row-major (so I[K*i + j] gives the j'th nearest neighbor of the i'th point)
         * @param D The euclidean distance array (true euclidean distance, not squared) (N_POINTSxK) row-major
         * @param N_DIM The number of dimensions of the input points
         * @param N_POINTS The number of input points
         * @param K The number of nearest neighbors to return. If >=1024, this function uses the CPU instead of the GPU
         */
        void knn(float* points, long* I, float* D, const unsigned int N_DIM, const unsigned int N_POINTS, const unsigned int K);

   }
#endif
