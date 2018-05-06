/**
 * @brief Utilities for doing math of different sorts
 * 
 * @file math_utils.h
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */


#ifndef SRC_INCLUDE_UTIL_MATH_UTILS_H_
#define SRC_INCLUDE_UTIL_MATH_UTILS_H_

#include "include/common.h"
#include "include/util/matrix_broadcast_utils.h"
#include "include/util/reduce_utils.h"

namespace tsne {
namespace util {

/**
* @brief Normalize a vector to have mean zero and variance one. The dimensions are normalized independently.
* 
* @param handle CUBLAS handle reference
* @param points The points to normalize (Column-Major NxNDIM matrix)
* @param num_points The number of points
* @param num_dims The dimension of each point
*/
void GaussianNormalizeDeviceVector(cublasHandle_t &handle,
        thrust::device_vector<float> &d_points,
        const uint32_t num_points, const uint32_t num_dims);

/**
* @brief Square a vector element-wise
* 
* @param d_input The vector to square
* @param d_out The output vector
*/
void SquareDeviceVector(thrust::device_vector<float> &d_out,
        const thrust::device_vector<float> &d_input);

/**
* @brief Take the square root of a vector element-wise
* 
* @param d_input The vector to square root
* @param d_out The output vector
*/
void SqrtDeviceVector(thrust::device_vector<float> &d_out,
        const thrust::device_vector<float> &d_input);

/**
 * @brief Compute the L2 Norm of a device vector
 * 
 * @param d_vector The vector to compute the norm of
 * @return float The L2 Norm
 */
float L2NormDeviceVector(const thrust::device_vector<float> &d_vector);

/**
 * @brief Check if any elements in the vector are NaN or Inf
 * 
 * @param d_vector The device vector to check
 * @return true Nan/Inf present
 * @return false No Nan/Inf present
 */
bool AnyNanOrInfDeviceVector(const thrust::device_vector<float> &d_vector);

/**
 * @brief Max-normalize a device vector in place
 * 
 * @param d_vector The vector to normalize
 */
void MaxNormalizeDeviceVector(thrust::device_vector<float> &d_vector);


/**
 * @brief Symmetrize the Pij matrix in CSR format
 * 
 * @param handle The CUSPARSE handle
 * @param d_values The values of the sparse pij matrix
 * @param d_indices The indices of the sparse pij matrix
 * @param d_symmetrized_values The symmetrized values matrix
 * @param d_symmetrized_colind The symmetrized column indicies
 * @param d_symmetrized_rowptr The symmetrized row values
 * @param num_points The number of points
 * @param num_near_neighbors The number of nearest neighbors
 * @param magnitude_factor The normalization magnitude factor
 */
void SymmetrizeMatrix(cusparseHandle_t &handle,
        thrust::device_vector<float> &d_values,
        thrust::device_vector<int32_t> &d_indices,
        thrust::device_vector<float> &d_symmetrized_values,
        thrust::device_vector<int32_t> &d_symmetrized_colind,
        thrust::device_vector<int32_t> &d_symmetrized_rowptr,
        uint32_t num_points, uint32_t num_near_neighbors,
        float magnitude_factor);

}  // namespace util
}  // namespace tsne

#endif  // SRC_INCLUDE_UTIL_MATH_UTILS_H_
