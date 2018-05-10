/**
 * @brief Debugging Utilities
 * 
 * @file debug_utils.h
 * @author David chan
 * @date 2018-05-05
 * Copyright (c) 2018, Regents of the University of California
 */


#ifndef SRC_INCLUDE_UTIL_DEBUG_UTILS_H_
#define SRC_INCLUDE_UTIL_DEBUG_UTILS_H_

#include "include/common.h"

namespace tsnecuda {
namespace debug {

/**
 * @brief Print the NxM device matrix
 * 
 * @tparam T The type of the device matrix
 * @param d_matrix The NxM matrix to print
 * @param N The number of rows in the matrix
 * @param M The number of columns in the matrix
 */
template <typename T>
void PrintArray(const thrust::device_vector<T> &d_matrix,
    const int N, const int M);

}  // namespace debug
}  // namespace tsnecuda



#endif  // SRC_INCLUDE_UTIL_DEBUG_UTILS_H_
