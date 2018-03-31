#ifndef TSNE_UTILS_H
#define TNSE_UTILS_H

#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void add_row_vec(thrust::device_vector<float> matrix, thrust::device_vector<float> vector, const unsigned int N, const unsigned int M, const float alpha);
void mul_row_vec(thrust::device_vector<float> matrix, thrust::device_vector<float> vector, const unsigned int N, const unsigned int M, const float alpha);
void div_row_vec(thrust::device_vector<float> matrix, thrust::device_vector<float> vector, const unsigned int N, const unsigned int M, const float alpha);
thrust::device_vector<float> square(const thrust::device_vector<float> vec, const unsigned int N);
thrust::device_vector<float> sqrt(const thrust::device_vector<float> vec, const unsigned int N);
thrust::device_vector<float> reduce_alpha(cublasHandle_t &handle, const thrust::device_vector<float> matrix, const unsigned int N, const unsigned int M, float alpha, const int axis);
thrust::device_vector<float> reduce_mean(cublasHandle_t &handle, const thrust::device_vector<float> matrix, const unsigned int N, const unsigned int M, const int axis);
thrust::device_vector<float> reduce_sum(cublasHandle_t &handle, const thrust::device_vector<float> matrix, const unsigned int N, const unsigned int M, const int axis);
thrust::device_vector<float> pairwise_dist(cublasHandle_t &handle, const thrust::device_vector<float> points, const unsigned int N, const unsigned int NDIMS);
void gauss_normalize(cublasHandle_t &handle, thrust::device_vector<float> points, const unsigned int N, const unsigned int NDIMS);

#endif
