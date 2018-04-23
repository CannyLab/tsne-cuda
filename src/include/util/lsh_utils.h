/**
 * @brief Euclidean LSH implementation for fast K-NN computation
 * 
 * @file lsh_utils.h
 * @author David Chan
 * @date 2018-04-23
 */

#ifndef LSH_UTILS_H
#define LSH_UTILS_H

#include "common.h"

namespace LSH {

    /**
    * @brief Returns in knn_matrix a N_POINTSxK matrix containing the K nearest
    * neighbors within distance R (with high probability)
    * 
    * @param knn_matrix The output matrix (N_POINTS * K, HOST)
    * @param points The input points (N_POINTS * N_DIM, HOST)
    * @param N_POINTS The number of points we're handling
    * @param N_DIM The number of dimensions
    * @param K The number of nearest neighbors to cap at
    * @param R The radius cutoff
    */
    void compute_knns(float* knn_matrix, 
                    float* points, 
                    const unsigned int N_POINTS, 
                    const unsigned int N_DIM, 
                    const unsigned int K,
                    const float R);
}




#endif