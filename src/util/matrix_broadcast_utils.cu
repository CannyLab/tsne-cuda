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
__global__ void Broadcast::_broadcast_row_vec(
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
__global__ void Broadcast::_broadcast_col_vec(
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

 template<typename BinaryFunction, typename T>
 void Broadcast::broadcast_matrix_vector(
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
        Broadcast::_broadcast_col_vec<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(matrix.data()),
                                                     thrust::raw_pointer_cast(vector.data()), 
                                                     N, M, binary_op, alpha);
     } else {
        Broadcast::_broadcast_row_vec<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(matrix.data()),
                                                     thrust::raw_pointer_cast(vector.data()), 
                                                     N, M, binary_op, alpha);
     }
 }


 // Explicit instantiations of the method
 template void Broadcast::broadcast_matrix_vector<thrust::divides<float>, float>(
    thrust::device_vector<float> &matrix, 
    const thrust::device_vector<float> &vector, 
    const unsigned int N, 
    const unsigned int M, 
    thrust::divides<float> binary_op,
    const unsigned int axis,
    const float alpha);
 template void Broadcast::broadcast_matrix_vector<thrust::minus<float>, float>(
    thrust::device_vector<float> &matrix, 
    const thrust::device_vector<float> &vector, 
    const unsigned int N, 
    const unsigned int M, 
    thrust::minus<float> binary_op,
    const unsigned int axis,
    const float alpha);
