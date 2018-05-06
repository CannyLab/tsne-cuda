/**
 * @brief Utilities for working with data/files
 * 
 * @file data_utils.h
 * @author Forrest Huang
 * @date 2018-05-05
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_DATA_UTILS_H_
#define SRC_INCLUDE_UTIL_DATA_UTILS_H_


// CXX Includes
#include <string>

// Local Includes
#include "include/common.h"


namespace tsne {
namespace util {

// TODO(David): Bad Practice to return float* of allocated memory. Make the
// Calling function responsible.

/**
 * @brief Load the MNIST data file.
 * 
 * @param file_name The name of the ubyte file to load
 * @param num_images The number of images that are loaded
 * @param num_rows The number of rows that are loaded
 * @param num_columns The number of columns that are loaded
 * @return float* A pointer to the loaded data
 */
float* LoadMnist(std::string file_name, int32_t& num_images,
    int32_t& num_rows, int32_t& num_columns);

/**
 * @brief Load the CIFAR-10 dataset
 * 
 * @param file_path The path to the root containing the bin files.
 * @return float* A pointer to the loaded data
 */
float* LoadCifar10(std::string file_path);

/**
 * @brief Load the CIFAR-100 dataset
 * 
 * @param file_name The file name of the cifar100 bin file. 
 * @return float* A pointer to the loaded data
 */
float* LoadCifar100(std::string file_name);

/**
 * @brief Save an array of points to the disk
 * 
 * @param points The array of points to save
 * @param file_name The file name to save the points to
 * @param num_points The number of points in the array
 * @param num_dims The number of dimensions in the array
 */
void Save(const float * const points, std::string file_name,
    const uint32_t num_points, const uint32_t num_dims);

/**
 * @brief Save a device vector to disk
 * 
 * @param d_points The device vector which we want to save
 * @param file_name The file name to save the points to
 * @param num_points The number of points in the array
 * @param num_dims The number of dimensions of points in the array
 */
void Save(thrust::device_vector<float> d_points, std::string file_name,
    const uint32_t num_points, const uint32_t num_dims);
float* Load(std::string file_name);

}  // namespace util
}  // namespace tsne


#endif  // SRC_INCLUDE_UTIL_DATA_UTILS_H_
