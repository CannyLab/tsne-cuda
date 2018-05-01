/**
 * @brief Utilities for doing math of different sorts
 * 
 * @file math_utils.h
 * @author David Chan
 * @date 2018-04-04
 */


#ifndef MATH_UTILS_H
#define MATH_UTILS_H

    #include "common.h"
    #include "util/matrix_broadcast_utils.h"
    #include "util/reduce_utils.h"

    namespace Math {
        /**
         * @brief Normalize a vector to have mean zero and variance one. The dimensions are normalized independently.
         * 
         * @param handle CUBLAS handle reference
         * @param points The points to normalize (Column-Major NxNDIM matrix)
         * @param N The number of points
         * @param NDIMS The dimension of each point
         */
        void gauss_normalize(cublasHandle_t &handle, thrust::device_vector<float> &points, const unsigned int N, const unsigned int NDIMS);

        /**
         * @brief Square a vector element-wise
         * 
         * @param vec The vector to square
         * @param out The output vector
         */
        void square(const thrust::device_vector<float> &vec, thrust::device_vector<float> &out);

        /**
         * @brief Take the square root of a vector element-wise
         * 
         * @param vec The vector to square root
         * @param out The output vector
         */
        void sqrt(const thrust::device_vector<float> &vec, thrust::device_vector<float> &out);
        
        /**
         * @brief Compute the L2 squared norm of a vector
         * 
         * @param vec The vector to compute
         * @return float The squared norm of vec
         */
        float norm(const thrust::device_vector<float> &vec);

        /**
         * @brief Checks if any element of a vector is NaN or inf.
         * 
         * @param vec The vector to check
         * @return bool If the vector contains NaNs of Infs 
         */
        bool any_nan_or_inf(const thrust::device_vector<float> &vec);

        /**
         * @brief Normalizes a vector by the maximum in-place
         * 
         * @param vec The vector to normalize
         */
        void max_norm(thrust::device_vector<float> &vec);
    }

    namespace Sparse {

        /**
         * @brief Symmetrize a sparse-built pij matrix by taking the sum and dividing. Note that all of the
         * sym_* values should be declared but not allocated with the exception of sym_nnz which shhould be the
         * address of an integer to return into (probably allocated on the stack). 
         * 
         * @param values The values of the FAISS-constructed pij matrix to symmetrize (DEVICE FLOAT MATRIX)
         * @param indices The indices of the FAISS matrix to symmetrize (DEVICE INDICES MATRIX)
         * @param sym_values DO NOT ALLOCATE (the function does this for you) The returned sparse matrix values (CSR format, DEVICE)
         * @param sym_colind DO NOT ALLOCATE (the function does this for you) The returned sparse matrix column indices (CSR format, DEVICE)
         * @param sym_rowptr DO NOT ALLOCATE (the function does this for you) The returned sparse matrix row pointers (CSR format, DEVICE)
         * @param sym_nnz The returned sparse matrix number of non-zeros (CSR format, HOST)
         * @param N_POINTS The number of points
         * @param K The number of nearest neighbors
         * @param magnitude_factor Floating point attractive force magnitude
         */
        void sym_mat_gpu(thrust::device_vector<float> &values, thrust::device_vector<int> &indices, thrust::device_vector<float> &sym_values,  
                                thrust::device_vector<int> &sym_colind, thrust::device_vector<int> &sym_rowptr, int* sym_nnz, 
                                unsigned int N_POINTS, unsigned int K, float magnitude_factor);

    }

#endif
