#ifndef __CUDA_UTILITY_H_
#define __CUDA_UTILITY_H_

#include <assert.h>
#include "defs.h"

static dim3 makeGrid(int nBlocks)
{
	int dim_x, dim_y, dim_z;
	
	if(nBlocks < 1)
		nBlocks = 1;
		
	if(nBlocks < MAX_BLOCKS_PER_DIMENSION)
		return dim3(nBlocks, 1, 1);
		
	float dimf = sqrtf((float)nBlocks);
	float power2 = ceilf(logf(dimf) / logf(2));
	
	dim_x = (int)powf(2.0f, power2);
	dim_y = (int)ceilf((nBlocks / (float)dim_x));
	dim_z = 1;
	
	while(dim_x >= MAX_BLOCKS_PER_DIMENSION)
	{
		if((dim_y * 2) < MAX_BLOCKS_PER_DIMENSION)
		{
			dim_y *= 2;
		}
		else
			dim_z *= 2;
			
		dim_x /= 2;
	}
	
	assert(dim_x*dim_y*dim_z >= nBlocks);
	return dim3(dim_x, dim_y, dim_z);
}


static __global__ void computePermutation(unsigned int nSamples, unsigned int *permutation)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nSamples)
		return;
		
	permutation[threadId] = threadId;
}

template <class T>
static __global__ void initMemory(T *data, unsigned int N, T initValue)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= N)
		return;
		
	data[threadId] = initValue;
}

#endif