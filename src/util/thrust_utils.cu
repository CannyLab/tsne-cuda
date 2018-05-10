/**
 * @brief Implementation of some thrust utility functions
 * 
 * @file thrust_utils.cu
 * @author David Chan
 * @date 2018-04-28
 * Copyright (c) 2018, Regents of the University of California
 */

#include "include/util/thrust_utils.h"

// Assumes that vec is an N x N matrix
void tsnecuda::util::ZeroDeviceMatrixDiagonal(
        thrust::device_vector<float> &d_vector, const int N) {
    typedef thrust::device_vector<float>::iterator Iterator;
    StridedRange<Iterator> return_vector(d_vector.begin(),
            d_vector.end(), N + 1);
    thrust::fill(return_vector.begin(), return_vector.end(), 0.0f);
}