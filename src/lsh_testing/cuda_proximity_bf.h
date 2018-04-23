#ifndef __CUDA_PROXIMITY_BF_H_
#define __CUDA_PROXIMITY_BF_H_

#include "cuda_metric.h"
#include "cuda_utility.h"
#include "radixsort.h"
#include "defs.h"
#include "cuda_defs.h"
#include "reduce_kernel.h"
#include <cutil.h>

//Brute-force proximity (min-dist is the threshold to remove almost the same neighbors, default 0)
template <typename T>
__global__ void proximityDistances(T* samples, unsigned int nSamples, T* queries, unsigned int nQueries, unsigned int nBegin, unsigned int nEnd, unsigned int dim, float* distances, T mindist)
{
	unsigned int M = nEnd - nBegin;
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nSamples * M)
		return;
		
		
	unsigned int queryId = threadId / nSamples;
	unsigned int setId = threadId - queryId * nSamples;
	queryId += nBegin;
	
	float dist = distance_sqr2_interleaved(queries + queryId, samples + setId, nQueries, nSamples, dim);
	if(dist <= mindist)
		dist = CUDART_NORM_HUGE_F;
	distances[threadId] = dist;
}

// brute force proximity computation using CUDA reduce
template <typename T>
void proximityComputation_bruteforce(T* samples, unsigned int nSamples, T* queries, unsigned int nQueries, unsigned int dim, unsigned int K, float mindist, unsigned int* KNNResult)
{
	unsigned int queryPerIter = min(1000, nQueries);
	
	float *distances = NULL;
	GPUMALLOC((void**)&distances, sizeof(float) * queryPerIter * nSamples);

	float logSize = ceil(log((float)nSamples) / log(2.0f));
	int filledSize = powf(2.0f, logSize);
	float* filledDistances = NULL;
	GPUMALLOC((void**)&filledDistances, sizeof(float) * filledSize);
	dim3 filledGrid = makeGrid((int)ceilf(filledSize / (float)PROXIMITY_THREADS));
	dim3 filledThreads = dim3(PROXIMITY_THREADS, 1, 1);
	initMemory<<<filledGrid, filledThreads>>>(filledDistances, filledSize, CUDART_NORM_HUGE_F);

	unsigned int iterNum = (unsigned int)ceil(nQueries / (float)queryPerIter);

	for(unsigned int i = 0; i < iterNum; ++i)
	{
		unsigned int nBegin = i * queryPerIter;
		unsigned int nEnd = (i + 1) * queryPerIter;
		if(nEnd > nQueries) nEnd = nQueries;

		dim3 distanceGrid = makeGrid((int)ceilf(nSamples * queryPerIter / (float)PROXIMITY_THREADS));
		dim3 distanceThreads = dim3(PROXIMITY_THREADS, 1, 1);

		proximityDistances <<< distanceGrid, distanceThreads >>> (samples, nSamples, queries, nQueries, nBegin, nEnd, dim, distances, mindist);

		float* curDistances = distances;
		for(unsigned int j = nBegin; j < nEnd; ++j)
		{
			GPUTOGPU(filledDistances, curDistances, sizeof(float) * nSamples);
			float resetMax = CUDART_NORM_HUGE_F;
			for(unsigned int k = 0; k < K; ++k)
			{
				float minValue = reduce_min(filledDistances, filledSize);
				unsigned int minId = get_elem_index(filledDistances, filledSize, minValue);
				TOGPU((filledDistances + minId), &resetMax, sizeof(float));
				TOGPU((KNNResult + k * nQueries + j), &minId, sizeof(unsigned int));
			}
			curDistances = curDistances + nSamples;
		}
	}

	GPUFREE(distances);
	GPUFREE(filledDistances);
}

__global__ void copyProximityResults(unsigned int *outputResult, unsigned int *tempOutputResult, unsigned int nSamples, unsigned int K)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nSamples * K)
		return;

	unsigned int sampleId = threadId / K;
	unsigned int KId = threadId - K * sampleId;

	outputResult[KId * nSamples + sampleId] = tempOutputResult[threadId];
}

// brute force proximity computation using radixsort
template <typename T>
void proximityComputation_bruteforce2(T* samples, unsigned int nSamples, T* queries, unsigned int nQueries, unsigned int dim, unsigned int K, T mindist, unsigned int* KNNResult)
{
	dim3 permutationGrid = makeGrid((int)ceilf(nSamples / (float)PROXIMITY_THREADS));
	dim3 permutationThreads = dim3(PROXIMITY_THREADS, 1, 1);

	unsigned int *permutation = NULL;
	GPUMALLOC((void**)&permutation, sizeof(unsigned int) * nSamples);
	unsigned int *permutation_copy = NULL;
	GPUMALLOC((void**)&permutation_copy, sizeof(unsigned int) * nSamples);

	computePermutation <<< permutationGrid, permutationThreads >>> (nSamples, permutation);

	unsigned int queryPerIter = min(1000, nQueries);

	float *distances = NULL;
	GPUMALLOC((void**)&distances, sizeof(float) * queryPerIter * nSamples);

	unsigned int iterNum = (unsigned int)ceilf(nQueries / (float)queryPerIter);
	nvRadixSort::RadixSort proximityRadixSort(nSamples);

	unsigned int *tempOutputResult = NULL;
	GPUMALLOC((void**)&tempOutputResult, sizeof(unsigned int) * K * nQueries);

	unsigned int *curOutputResult = tempOutputResult;

	for(unsigned int i = 0; i < iterNum; ++i)
	{
		unsigned int nBegin = i * queryPerIter;
		unsigned int nEnd = (i + 1) * queryPerIter;
		if(nEnd > nQueries) nEnd = nQueries;

		dim3 distanceGrid = makeGrid((int)ceilf(nSamples * queryPerIter / (float)PROXIMITY_THREADS));
		dim3 distanceThreads = dim3(PROXIMITY_THREADS, 1, 1);

		proximityDistances <<< distanceGrid, distanceThreads >>> (samples, nSamples, queries, nQueries, nBegin, nEnd, dim, distances, mindist);

		float* curDistances = distances;
		for(unsigned int j = nBegin; j < nEnd; ++j)
		{
			GPUTOGPU(permutation_copy, permutation, sizeof(unsigned int) * nSamples);
			proximityRadixSort.sort(curDistances, permutation_copy, nSamples, sizeof(float) * 8, false);
			curDistances = curDistances + nSamples;
			GPUTOGPU(curOutputResult, permutation_copy, sizeof(unsigned int) * K);
			curOutputResult = curOutputResult + K;
		}
	}

	dim3 postCopyGrid = makeGrid((int)ceilf(nQueries * K / (float)PROXIMITY_THREADS));
	dim3 postCopyThreads = dim3(PROXIMITY_THREADS, 1, 1);
	copyProximityResults <<< postCopyGrid, postCopyThreads >>> (KNNResult, tempOutputResult, nQueries, K);

	GPUFREE(tempOutputResult);
	GPUFREE(distances);
	GPUFREE(permutation);
	GPUFREE(permutation_copy);
}

#endif