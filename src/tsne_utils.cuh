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

thrust::device_vector<float> pairwise_dist(const thrust::device_vector<float> data, const unsigned int N, const unsigned int NDIMS);

#endif
