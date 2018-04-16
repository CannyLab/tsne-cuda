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

    namespace Random {
            /**
            * @brief Returns a uniform-random float device vector in range [0,1]
            * 
            * @param N The length of the vector to return
            * @return thrust::device_vector<float> 
            */
            thrust::device_vector<float> random_vector(const unsigned int N);

            /**
             * @brief Returns a uniform-random float device vector in a given range
             * 
             * @param N The length of the vector to return
             * @param lb The lower bound of numbers to return
             * @param ub The upper bound of numbers to return
             * @return thrust::device_vector<float> 
             */
            thrust::device_vector<float> rand_in_range(const unsigned int N, float lb, float ub);

    }

#endif