/**
 * @brief Implementation file for matrix broadcasting
 * 
 * @file matrix_broadcast.cu
 * @author David Chan
 * @date 2018-04-04
 */

 #include "util/matrix_broadcast_utils.h"

 // Performs the operation matrix[i, :] = binary_op(matrix[i, :], alpha * vector) for each row i in the matrix
template<typename BinaryFunction, typename T>
__global__ void _broadcast_row_vec(
                                    T * __restrict__ matrix, 
                                    const T * __restrict__ vector, 
                                    const unsigned int N, 
                                    const unsigned int M, 
                                    BinaryFunction binary_op, 
                                    const T alpha) 
{
    const unsigned int TID = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int i = TID % N;
    const unsigned int j = TID / N;

    if (j < M) matrix[j * N + i] = binary_op(matrix[j * N + i], alpha * vector[j]);
}

// Performs the operation matrix[:, j] = binary_op(matrix[:, j], alpha * vector) for each col i in the matrix
template<typename BinaryFunction, typename T>
__global__ void _broadcast_col_vec(
                                    T * __restrict__ matrix, 
                                    const T * __restrict__ vector, 
                                    const unsigned int N, 
                                    const unsigned int M,
                                    BinaryFunction binary_op,
                                    const T alpha)
{
     const unsigned int TID = threadIdx.x + blockIdx.x * blockDim.x;
     const unsigned int i = TID % N;
     const unsigned int j = TID / N;

     if (j < M) matrix[j * N + i] = binary_op(matrix[j * N + i], alpha * vector[i]);
}

 /**
  * @brief 
  * 
  * @tparam BinaryFunction 
  * @tparam T Matrix format
  * @param matrix (N x M) matrix stored in column major order
  * @param vector Length N vector if axis == 0, length M vector if axis == 1
  * @param N,M dimensions of matrix
  * @param binary_op an operation that takes in two arguments of type T and returns a type T
  * @param axis 0 or 1, controlls whether this runs a column or row broadcast
  * @param alpha scalar multiple for vector
  * 
  * @note 
  * should axis == 0 be row or column broadcasting? and vice versa for axis == 1?
  *
  *
  */
 template<typename BinaryFunction, typename T>
 void broadcast_matrix_vector(
                             thrust::device_vector<T> &matrix, 
                             const thrust::device_vector<T> &vector, 
                             const unsigned int N, 
                             const unsigned int M, 
                             BinaryFunction binary_op,
                             const unsigned int axis,
                             const T alpha) 
 { 
     // Checks to make sure dimensions are correct
     assert(matrix.size() >= N * M);
     assert((axis == 0 && vector.size() >= N) || (axis == 1 && vector.size() >= M));
     
     const unsigned int BLOCKSIZE = 32;
     const unsigned int NBLOCKS = iDivUp(N * M, BLOCKSIZE);
     if (axis == 0) {
         _broadcast_col_vec<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(matrix.data()),
                                                     thrust::raw_pointer_cast(vector.data()), 
                                                     N, M, binary_op, alpha);
     } else {
         _broadcast_row_vec<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(matrix.data()),
                                                     thrust::raw_pointer_cast(vector.data()), 
                                                     N, M, binary_op, alpha);
     }
 }
