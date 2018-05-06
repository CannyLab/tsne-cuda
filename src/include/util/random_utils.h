/**
 * @brief Utilities for handling random numbers
 * 
 * @file random_utils.h
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_RANDOM_UTILS_H_
#define SRC_INCLUDE_UTIL_RANDOM_UTILS_H_

#include "include/common.h"
#include "include/util/cuda_utils.h"

namespace tsne {
namespace util {

/**
* @brief Returns a uniform-random float device vector in range [0,1]
* 
* @param vector_size The length of the vector to return
* @return thrust::device_vector<float> 
*/
thrust::device_vector<float> RandomDeviceUniformZeroOneVector(
        const uint32 vector_size);

/**
 * @brief Returns a uniform random device vector in the specified range
 * 
 * @param vector_size The size of the device vector to return
 * @param lower_bound The lower bound
 * @param upper_bound The upper bound
 * @return thrust::device_vector<float> The random vector
 */
thrust::device_vector<float> RandomDeviceVectorInRange(const uint32 vector_size,
        float lower_bound, float upper_bound);

}
}

#endif  // SRC_INCLUDE_UTIL_RANDOM_UTILS_H_
