/**
 * @brief Naive implementation of T-SNE O(n^2) on CPU
 * 
 * @file naive_tsne_cpu.h
 * @author David Chan
 * @date 2018-04-04
 */

#ifndef NAIVE_TSNE_CPU_H
#define NAIVE_TSNE_CPU_H

#include "common.h"
#include "util/cuda_utils.h"
#include "util/math_utils.h"
#include "util/matrix_broadcast_utils.h"
#include "util/reduce_utils.h"
#include "util/distance_utils.h"
#include "util/random_utils.h"

std::vector<float> compute_pij_cpu(std::vector<float> &points, 
	                           std::vector<float> &sigma, 
	                           const unsigned int N, 
	                           const unsigned int NDIMS);

/*
std::vector<float> compute_qij_cpu(std::vector<float> &ys,
                                   const unsigned int N,
                                   const unsigned int PROJDIM);

float compute_gradients_cpu(std::vector<float> &forces,
                        std::vector<float> &dist, 
                        std::vector<float> &ys, 
                        std::vector<float> &pij, 
                        std::vector<float> &qij, 
                        const unsigned int N,
                        float eta);

std::vector<float> naive_tsne_cpu(std::vector<float> &points, 
                              const unsigned int N, 
                              const unsigned int NDIMS);
*/
#endif
