#ifndef __CUDA_PRNG_RAND_H_
#define __CUDA_PRNG_RAND_H_

//////////////////////////////////////////////////////////////////////////////////////
// PRNG randome generator (GPU gems 3)
//////////////////////////////////////////////////////////////////////////////////////


#include <cutil.h>
#define RAND_BLOCK_SIZE 256

extern "C"
__device__ unsigned LCGStep(unsigned &z, unsigned A, unsigned C)
{
	return z = (A * z + C);
}

extern "C"
__device__ unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
	unsigned b = (((z << S1) ^ z) >> S2);
	return z = (((z & M) << S3) ^ b);
}

extern "C"
__device__ float HybridTaus(unsigned &z1, unsigned &z2, unsigned &z3, unsigned &z4)
{
	// Combined period is lcm(p1,p2,p3,p4) ~ 2^121
	return 2.3283064365387e-10f * (              // Periods
	           TausStep(z1, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1
	           TausStep(z2, 2, 25, 4, 4294967288UL) ^    // p2=2^30-1
	           TausStep(z3, 3, 11, 17, 4294967280UL) ^   // p3=2^28-1
	           LCGStep(z4, 1664525, 1013904223UL)        // p4=2^32
	       );
}

extern "C"
__global__ void generate_uniform_(float *pOutput, int nDim, uint4 *pSeedArray, int nRand)
{
	int tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if(tid < nRand)
	{
		uint4* pSeed = pSeedArray + tid;
		float* pCurrent = pOutput + nDim * tid;
		
		float sample;
		
		for(int i = 0; i < nDim; ++i)
		{
			sample = HybridTaus(pSeed->x, pSeed->y, pSeed->z, pSeed->w);
			pCurrent[i] = sample;
		}
	}
}

extern "C"
void generate_uniform(int nDim, int nRand)
{
	float* phOutput = new float[nDim * nRand];
	uint4 *phSeed = new uint4[nRand];
	
	float* pdOutput;
	uint4 *pdSeed;
	
	GPUMALLOC((void**)&pdOutput, sizeof(float) * nDim * nRand);
	GPUMALLOC((void**)&pdSeed, sizeof(uint4) * nRand);
	
	srand(1);
	uint4 *pSeed = phSeed;
	for(int i = 0; i < nRand; ++i)
	{
		pSeed->x = rand() + 128;
		pSeed->y = rand() + 128;
		pSeed->z = rand() + 128;
		pSeed->w = rand();
		pSeed++;
	}
	
	TOGPU(pdSeed, phSeed, sizeof(uint4) * nRand);
	
	int nBlocks = (int)ceilf(nRand / (float)RAND_BLOCK_SIZE);
	dim3 grid(nBlocks, 1, 1);
	dim3 threads(RAND_BLOCK_SIZE, 1, 1);
	generate_uniform_ <<< grid, threads>>>(pdOutput, nDim, pdSeed, nRand);
	
	FROMGPU(phOutput, pdOutput, sizeof(float) * nDim * nRand);
	
	for(int i = 0; i < nRand; ++i)
	{
		for(int j = 0; j < nDim; ++j)
		{
			printf("%f ", phOutput[i * nDim + j]);
		}
		printf("\n");
	}
	
	GPUFREE(pdOutput);
	GPUFREE(pdSeed);
	
	delete [] phSeed; phSeed = NULL;
	delete [] phOutput; phOutput = NULL;
}


#endif