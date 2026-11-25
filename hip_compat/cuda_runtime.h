// HIP build only: maps the CUDA runtime names used by tsnecuda to HIP.
#ifndef TSNECUDA_HIP_COMPAT_CUDA_RUNTIME_H
#define TSNECUDA_HIP_COMPAT_CUDA_RUNTIME_H

#include <hip/hip_runtime.h>

#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemsetAsync hipMemsetAsync
#define cudaSetDevice hipSetDevice
#define cudaSuccess hipSuccess

#endif
