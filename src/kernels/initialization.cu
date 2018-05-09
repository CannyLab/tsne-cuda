/*
Kernel to initialize the global variables
*/

#include "kernels/include/initialization.h"


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

void tsnecuda::bh::Initialize(int * __restrict__ errd) 
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
    
    tsnecuda::bh::InitializationKernel<<<1, 1>>>(thrust::raw_pointer_cast(errl.data()));
    GpuErrorCheck(cudaDeviceSynchronize());
}

void tsnecuda::naive::Initialize() 
{
    // TODO: Add cache config sets for naive
}
