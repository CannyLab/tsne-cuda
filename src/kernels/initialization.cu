/*
Kernel to initialize the global variables
*/

#include "include/kernels/initialization.h"

__device__ volatile int stepd, bottomd, maxdepthd;
__device__ unsigned int blkcntd;
__device__ volatile float radiusd;

/******************************************************************************/
/*** initialize memory ********************************************************/
/******************************************************************************/

__global__ void tsnecuda::bh::InitializationKernel(int * __restrict errd)
{
    *errd = 0;
    stepd = -1;
    maxdepthd = 1;
    blkcntd = 0;
}

void tsnecuda::bh::Initialize(tsnecuda::GpuOptions &gpu_opt, thrust::device_vector<int> &errd) 
{
    cudaFuncSetCacheConfig(tsnecuda::bh::BoundingBoxKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(tsnecuda::bh::TreeBuildingKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(tsnecuda::bh::ClearKernel1, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(tsnecuda::bh::ClearKernel2, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(tsnecuda::bh::SummarizationKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(tsnecuda::bh::SortKernel, cudaFuncCachePreferL1);
    #ifdef __KEPLER__
    cudaFuncSetCacheConfig(tsnecuda::bh::ForceCalculationKernel, cudaFuncCachePreferEqual);
    #else
    cudaFuncSetCacheConfig(tsnecuda::bh::ForceCalculationKernel, cudaFuncCachePreferL1);
    #endif
    cudaFuncSetCacheConfig(tsnecuda::bh::IntegrationKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(tsnecuda::bh::ComputePijxQijKernel, cudaFuncCachePreferShared);
    
    tsnecuda::bh::InitializationKernel<<<1, 1>>>(thrust::raw_pointer_cast(errd.data()));
    GpuErrorCheck(cudaDeviceSynchronize());
}

void tsnecuda::naive::Initialize() 
{
    // TODO: Add cache config sets for naive
}
