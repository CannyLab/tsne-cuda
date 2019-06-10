#include "../include/kernels/initialization.h"

/******************************************************************************/
/*** initialize memory ********************************************************/
/******************************************************************************/


void tsnecuda::Initialize(tsnecuda::GpuOptions &gpu_opt, thrust::device_vector<int> &errd)
{
    cudaFuncSetCacheConfig(tsnecuda::bh::IntegrationKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(tsnecuda::bh::ComputePijxQijKernel, cudaFuncCachePreferShared);
    GpuErrorCheck(cudaDeviceSynchronize());
}
