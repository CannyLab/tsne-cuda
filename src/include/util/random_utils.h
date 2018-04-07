/**
 * @brief Utilities for handling random numbers
 * 
 * @file random_utils.h
 * @author David Chan
 * @date 2018-04-04
 */

#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

    #include "common.h"
	#include "util/cuda_utils.h"

    /**
    * @brief Returns a uniform-random float device vector
    * 
    * @param N The length of the vector to return
    * @return thrust::device_vector<float> 
    */
    thrust::device_vector<float> random_vector(const unsigned int N);
    thrust::device_vector<float> rand_in_range(const unsigned int N, float lb, float ub);

#endif