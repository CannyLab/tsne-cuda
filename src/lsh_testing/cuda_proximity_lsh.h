#ifndef __CUDA_PROXIMITY_LSH_H_
#define __CUDA_PROXIMITY_LSH_H_


#include "cuda_hash.h"
#include "cuda_heap.h"
#include "cuda_timer.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

__device__ __constant__ float constant_b[LSH_M];
__device__ __constant__ int constant_P[LSH_nP];
__device__ __constant__ float constant_A[LSH_M * 32]; // try to change it for different dimension (here maximum is 32)

float gaussian_rand(float mean, float sigma)
{
	const float norm = 1.0 / (RAND_MAX + 1.0);
	float u = 1.0f - rand() * norm;
	float v = rand() * norm;
	float z = sqrt(-2.0f * log(u)) * cos(2.0f * 3.14159265358979323846f * v);
	return mean + sigma * z;
}

__device__ __inline__ unsigned int simple_rand(unsigned int seed)
{
	unsigned long next = seed;
	next = next * 1103515245 + 12345;
	next = next / 65536;
	unsigned int t = (unsigned int)next / 32768;
	next -= t * 32768;
	return next;
}

// 0 <= m < M
__device__ __inline__ int LSH_hashing(float* data, unsigned int m, unsigned int nSamples, unsigned int dim, float W)
{
	float tmp = 0;
	for(int i = 0; i < dim; ++i)
	{
		tmp += (constant_A[m * dim + i] * data[i * nSamples]);
	}
	tmp += constant_b[m];
	tmp /= W;
	return floorf(tmp);
}

__device__ __inline__ unsigned int LSH_hashing_hashed(float* data, unsigned int nSamples, unsigned int dim, float W)
{
	int key = 0;
	for(int i = 0; i < LSH_M;)
	{
		int pos = i / 2;
		key += LSH_hashing(data, pos, nSamples, dim, W) * constant_P[pos];
		i++;
		
		if(i >= LSH_M) break;
		
		pos = LSH_M - (i + 1) / 2;
		key += LSH_hashing(data, pos, nSamples, dim, W) * constant_P[pos];
		i++;
	}
	
	return key;
}

__device__ __inline__ unsigned int LSH_hashing_diff(float* samples, unsigned int id1, unsigned int id2, unsigned int nSamples, unsigned int dim, float W)
{
	unsigned int diff = 0;
	unsigned int n = id2 / nSamples;
	id2 -= n * nSamples;
	
	for(int i = 0; i < LSH_M; ++i)
	{
		diff += (LSH_hashing(samples + id1 * dim, i, nSamples, dim, W) - LSH_hashing(samples + id2 * dim, i, nSamples, dim, W));
	}
	return diff;
}

// samples: #nSamples
// hashvalues: nSamples
// nSamples is the number of samples
// M is the number of hash functions in LSH (g_i(x) = {h_1(x), ..., h_M(x)}
// L is the number of LSH tables
// d_P is the random parameters designed for hashing g_i(x) (hash(g_i(x))
// nP is the size of n_P
__global__ void LSHHashComputation(float* samples, unsigned int* hashvalues, unsigned int nSamples, unsigned int dim, float W)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nSamples)
		return;
		
	hashvalues[threadId] = LSH_hashing_hashed(samples + threadId, nSamples, dim, W);
}

__global__ void gpu_diff(unsigned int* v, unsigned int* diff, unsigned int n)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= n)
		return;
		
	if(threadId == 0)
		diff[threadId] = 1;
	else
	{
		if(v[threadId] == v[threadId - 1])
			diff[threadId] = 0;
		else
			diff[threadId] = 1;
	}
}

__global__ void gpu_compute_unique(unsigned int* diff, unsigned int* keys, unsigned int* diff_sum, unsigned int* unique_starts, unsigned int* unique_keys, unsigned int n)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= n)
		return;
		
	if(diff[threadId] == 1)
	{
		unique_starts[diff_sum[threadId] - 1] = threadId;
		unique_keys[diff_sum[threadId] - 1] = keys[threadId];
	}
}

__global__ void gpu_compute_unique_starts(unsigned int* unique_starts, unsigned int* unique_counts, unsigned int nUnique, unsigned int n)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nUnique)
		return;
		
	if(threadId == nUnique - 1)
		unique_counts[threadId] = n - unique_starts[threadId];
	else
		unique_counts[threadId] = unique_starts[threadId + 1] - unique_starts[threadId];
}

__global__ void gpu_find_unused_key(unsigned int* keys, unsigned int n, unsigned int* unused)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= n)
		return;
		
	if(threadId > 1)
	{
		if(keys[threadId] != keys[threadId - 1] + 1)
			atomicExch(unused, keys[threadId] - 1);
	}
}

__global__ void gpu_init_maxheaps(CUDA_MaxHeap_Interleaved* heaps, unsigned int* proximityResults, float* proximityDistances, unsigned int nSamples, unsigned int nProximitySize)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nSamples)
		return;
		
	heaps[threadId].list = (void*)(proximityDistances + threadId);
	heaps[threadId].list_assoc = (void*)(proximityResults + threadId);
	heaps[threadId].size = 0;
}


// change the distribution of result
// change the order
__global__ void gpu_hashing_postprocess(CUDA_MaxHeap_Interleaved* heaps, unsigned int nHeaps, unsigned int K, unsigned int* outputResults, float* outputDistances)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nHeaps)
		return;
		
	CUDA_MaxHeap_Interleaved* heap = &heaps[threadId];
	int heap_size = heap->size;
	float dist;
	unsigned int id;
	
	if(outputDistances == NULL)
	{
		for(int i = K - 1; i >= 0; i--)
		{
			if(i < heap_size)
			{
				extract_max<float, unsigned int>(heap, &dist, &id, nHeaps);
				outputResults[nHeaps * i + threadId] = id;
			}
			else
			{
				outputResults[nHeaps * i + threadId] = -1;
			}
		}
	}
	else
	{
		for(int i = K - 1; i >= 0; i--)
		{
			if(i < heap_size)
			{
				extract_max<float, unsigned int>(heap, &dist, &id, nHeaps);
				outputResults[nHeaps * i + threadId] = id;
				outputDistances[nHeaps * i + threadId] = dist;
			}
			else
			{
				outputResults[nHeaps * i + threadId] = -1;
				outputDistances[nHeaps * i + threadId] = CUDART_NORM_HUGE_F;
			}
		}
	}
}


__global__ void LSHProximity(float *samples, float* queries, unsigned int* LSHHashIndices,
                             unsigned int* uniqueKeys_starts, unsigned int* uniqueKeys_counts,
                             unsigned int* hashTable, unsigned int* hashTableValue,
                             CUDA_MaxHeap_Interleaved* heaps, unsigned int maxHeapSize,
                             unsigned int nUniqueKeys, unsigned int nSamples, unsigned int nQueries,
                             unsigned int dim, float mindist,
                             float W,
                             int c0, int c1,
                             int* d_debug)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nQueries)
		return;
		
	CUDA_MaxHeap_Interleaved* heap = &heaps[threadId];
	float* list = (float*)heap->list;
	unsigned int* list_assoc = (unsigned int*)heap->list_assoc;
	
	float* data = queries + threadId;
	
	unsigned int start, count;
	{
		unsigned int knnLSHHashIndices = -1;
		unsigned int LSHHashKey = LSH_hashing_hashed(data, nQueries, dim, W);
		unsigned int nBuckets = ceilf((float)nUniqueKeys / CUCKOO_HASH_PARAMETER);
		unsigned int bucketId = findBucketPerKey(LSHHashKey, c0, c1, nBuckets);

	
		for(int i = 0; i < CUCKOO_TABLE_NUM; ++i)
		{
			unsigned int offset = compute_cuckoo_hash_value(LSHHashKey, constant_c0s[i], constant_c1s[i]);
			offset += (i * nBuckets + bucketId) * CUCKOO_HASH_PER_BLOCK_SIZE;
		
			if(hashTable[offset] == LSHHashKey)
			{
				knnLSHHashIndices = hashTableValue[offset];
			}
		}
	
		if(knnLSHHashIndices == -1) return;

		start = uniqueKeys_starts[knnLSHHashIndices];
		count = uniqueKeys_counts[knnLSHHashIndices];
	}
	
	if(count <= LSH_MAXSORT)
	{
		for(int i = 0; i < count; ++i)
		{
			int id = LSHHashIndices[start + i];
			
			//unsigned int diff = LSH_hashing_diff(samples, threadId, id, nSamples, dim, W);
			//if(diff >= 4)
			//	continue;
			
			{
				// make sure no repeated item
				bool repeated = false;
				for(int j = 0; j < heap->size; ++j)
				{
					if(list_assoc[j * nQueries] == id)
					{
						repeated = true;
						break;
					}
				}
				if(repeated) continue;
			}

			float dist = distance_sqr2_interleaved(data, samples + id, nQueries, nSamples, dim);

			if(dist <= mindist) continue;
			
			if(heap->size < maxHeapSize)
			{
				insert<float, unsigned int>(heap, dist, id, nQueries);
			}
			else
			{
				if(dist < list[0])
				{
					list[0] = dist;
					list_assoc[0] = id;
					max_heapify<float, unsigned int>(heap, 0, nQueries);
				}
			}
		}
	}
	else
	{
		unsigned int next = threadId;
		for(int i = 0; i < LSH_MAXSORT; ++i)
		{
			next = simple_rand(next);			
			int id = next % count;
			id = LSHHashIndices[start + id];
			
			//unsigned int diff = LSH_hashing_diff(samples, threadId, id, nSamples, dim, W);
			//if(diff >= 4)
			//	continue;
							
			{
				// make sure no repeated item
				bool repeated = false;
				for(int j = 0; j < heap->size; ++j)
				{
					if(list_assoc[j * nQueries] == id)
					{
						repeated = true;
						break;
					}
				}
				if(repeated) continue;
			}

			
			float dist = distance_sqr2_interleaved(data, samples + id, nQueries, nSamples, dim);
			if(dist <= mindist) continue;
			
			if(heap->size < maxHeapSize)
			{
				insert<float, unsigned int>(heap, dist, id, nQueries);
			}
			else
			{
				if(dist < list[0])
				{
					list[0] = dist;
					list_assoc[0] = id;
					max_heapify<float, unsigned int>(heap, 0, nQueries);
				}
			}
		}
	}
	
}

__global__ void elementGrouping_phase1(unsigned int* elements, unsigned int* hashTable, unsigned int* counts, unsigned int* localOffsets, int c0, int c1, unsigned int nElements, unsigned int hashTableSize)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nElements)
		return;

	unsigned int key = elements[threadId];
	unsigned int nBuckets = ceilf((float)nElements / CUCKOO_HASH_PARAMETER);
	unsigned int bucketId = findBucketPerKey(key, c0, c1, nBuckets);


	for(int i = 0; i < CUCKOO_TABLE_NUM; ++i)
	{
		unsigned int offset = compute_cuckoo_hash_value(key, constant_c0s[i], constant_c1s[i]);
		offset += (i * nBuckets + bucketId) * CUCKOO_HASH_PER_BLOCK_SIZE;

		if(hashTable[offset] == key)
		{
			unsigned int* count = counts + offset;
			localOffsets[threadId] = atomicAdd(count, 1);
			break;
		}
	}
}

__global__ void elementGrouping_phase2(unsigned int* elements, unsigned int* hashTable, unsigned int* starts, unsigned int* localOffsets, unsigned int* rearrangedKeys, unsigned int* indices, int c0, int c1, unsigned int nElements, unsigned int hashTableSize)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nElements)
		return;

	unsigned int key = elements[threadId];
	unsigned int nBuckets = ceilf((float)nElements / CUCKOO_HASH_PARAMETER);
	unsigned int bucketId = findBucketPerKey(key, c0, c1, nBuckets);
	

	for(int i = 0; i < CUCKOO_TABLE_NUM; ++i)
	{
		unsigned int offset = compute_cuckoo_hash_value(key, constant_c0s[i], constant_c1s[i]);
		offset += (i * nBuckets + bucketId) * CUCKOO_HASH_PER_BLOCK_SIZE;

		if(hashTable[offset] == key)
		{
			unsigned int start = starts[offset];
			start += localOffsets[threadId];
			rearrangedKeys[start] = key;
			indices[start] = threadId;
			break;
		}
	}
}

void hashingBasedElementGrouping(unsigned int* hashValues, unsigned int* indices, unsigned int nSamples)
{
	int c0, c1;
	int* c0s = NULL;
	int* c1s = NULL;
	int* constants = NULL;
	c0s = new int[CUCKOO_TABLE_NUM];
	c1s = new int[CUCKOO_TABLE_NUM];
	constants = new int[CUCKOO_TABLE_NUM * 2];

	bool bIsConstructed = false;

	while(!bIsConstructed)
	{
		c0 = rand();
		c1 = rand();
		for(int k = 0; k < CUCKOO_TABLE_NUM * 2; ++k)
			constants[k] = rand();

		int ctmp = rand();
		for(int k = 0; k < CUCKOO_TABLE_NUM; ++k)
		{
			c0s[k] = ctmp ^ constants[2 * k];
			c1s[k] = ctmp ^ constants[2 * k + 1];
		}

		TOGPU_CONSTANT(constant_c0s, c0s, sizeof(int) * CUCKOO_TABLE_NUM, 0);
		TOGPU_CONSTANT(constant_c1s, c1s, sizeof(int) * CUCKOO_TABLE_NUM, 0);

		unsigned int* hashTable = NULL;
		unsigned int hashTableSize;
		bIsConstructed = constructHashTable(hashValues, nSamples, -1, c0, c1,
			hashTable, hashTableSize);

		if(bIsConstructed)
		{
			unsigned int* counts = NULL;
			unsigned int* starts = NULL;
			unsigned int* localOffsets = NULL;
			unsigned int* rearrangedKeys = NULL;
			GPUMALLOC((void**)&counts, sizeof(unsigned int) * hashTableSize);
			GPUMALLOC((void**)&starts, sizeof(unsigned int) * hashTableSize);
			GPUMEMSET(counts, 0, sizeof(unsigned int) * hashTableSize);
			GPUMALLOC((void**)&localOffsets, sizeof(unsigned int) * nSamples);
			GPUMALLOC((void**)&rearrangedKeys, sizeof(unsigned int) * nSamples);

			dim3 grid = makeGrid((int)ceilf(nSamples / (float)GENERAL_THREADS));
			dim3 threads = dim3(GENERAL_THREADS, 1, 1);

			elementGrouping_phase1 <<<grid, threads>>> (hashValues, hashTable, counts, localOffsets, c0, c1, nSamples, hashTableSize);
			cudaThreadSynchronize();

			CUDPPConfiguration config;
			CUDPPHandle scanPlan;
			config.algorithm = CUDPP_SCAN;
			config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
			config.op = CUDPP_ADD;
			config.datatype = CUDPP_INT;
			cudppPlan(&scanPlan, config, nSamples, 1, 0);
			cudppScan(scanPlan, starts, counts, nSamples);
			cudppDestroyPlan(scanPlan);

			cudaThreadSynchronize();

			GPUFREE(counts);

			elementGrouping_phase2 <<<grid, threads>>> (hashValues, hashTable, starts, localOffsets, rearrangedKeys, indices, c0, c1, nSamples, hashTableSize);

			GPUTOGPU(hashValues, rearrangedKeys, sizeof(unsigned int) * nSamples);


			GPUFREE(starts);
			GPUFREE(localOffsets);
			GPUFREE(rearrangedKeys);
		}


		GPUFREE(hashTable);
	}

	delete [] c0s;
	delete [] c1s;
	delete [] constants;

}


// samples are configurations
// nSamples is the number of samples
// K is the KNN parameter
// M is the number of hash functions in LSH (g_i(x) = {h_1(x), ..., h_M(x)}
// L is the number of LSH tables
// h_P is the random parameters designed for hashing g_i(x) (hash(g_i(x))
// n_P is the size of n_P
// h_upper and h_lower are configuration limits
// outputResult is KNN result
// for each sample, we compute its LSH hash g_i(x), i= 1... L. However, {g_i{x}}_{i=1:nSamples} is sparse, so we
// further compute the hash of g_i(x) and then store it in a cuckoo hash table.
// we use multi-probe method to compute the nearest neighbors.
// default L = 5
// default 	int h_P[10] = {1, 2, 5, 11, 17, 23, 31, 41, 47, 59}; and n_P = 10
// if (M > nP) then M will be set as nP automatically

void proximityComputation_LSH(float *samples, unsigned int nSamples, float* queries, unsigned int nQueries, unsigned int dim, unsigned int K, int L, float mindist, float *h_upper, float* h_lower, unsigned int *outputResult)
{
	//srand(1);
	dim3 grid = makeGrid((int)ceilf(nSamples / (float)PROXIMITY_THREADS));
	dim3 queryGrid = makeGrid((int)ceilf(nQueries / (float)PROXIMITY_THREADS));
	dim3 queryKGrid = makeGrid((int)ceilf(nQueries * K / (float)PROXIMITY_THREADS));
	dim3 threads = dim3(PROXIMITY_THREADS, 1, 1);
	
	int* h_P = NULL;
	CPUMALLOC((void**)&h_P, sizeof(int) * LSH_nP);
	for(int i = 0; i < LSH_nP; ++i)
		h_P[i] = rand();
		
	TOGPU_CONSTANT(constant_P, h_P, sizeof(int) * LSH_nP, 0);
	CPUFREE(h_P);
	
	unsigned int *proximityResults = NULL;
	float *proximityDistances = NULL;
	CUDA_MaxHeap_Interleaved* proximityMaxHeaps = NULL;
	GPUMALLOC((void**)&proximityResults, sizeof(unsigned int) * nQueries * K);
	GPUMALLOC((void**)&proximityDistances, sizeof(float) * nQueries * K);
	GPUMALLOC((void**)&proximityMaxHeaps, sizeof(CUDA_MaxHeap_Interleaved) * nQueries);
	
	initMemory<unsigned int> <<< queryKGrid, threads>>>(proximityResults, nQueries * K, -1);
	initMemory<float> <<< queryKGrid, threads>>>(proximityDistances, nQueries * K, CUDART_NORM_HUGE_F);
	gpu_init_maxheaps <<< queryGrid, threads>>>(proximityMaxHeaps, proximityResults, proximityDistances, nQueries, K);
	cudaThreadSynchronize();
	
	unsigned int D = dim;
	
	if(LSH_M > LSH_nP)
		printf("Warning: M > nP, so M will be set as nP automatically!\n");
		
	//compute W
	float* limits = new float[D];
	for(unsigned int i = 0; i < D; ++i)
	{
		if(fabs(h_upper[i]) > fabs(h_lower[i]))
			limits[i] = fabs(h_upper[i]);
		else
			limits[i] = fabs(h_lower[i]);
	}
	
	float rangeAct = 0;
	for(unsigned int i = 0; i < D; ++i)
		rangeAct += 2 * limits[i] * 2 * sqrt((float)D);
	float W = rangeAct / 16;
	
	delete [] limits; limits = NULL;
	
	float ratio = nSamples / 10000.0f; //4000.0
	ratio = powf(ratio, 1.0f / D);
	W /= ratio;
	
	printf("W %f\n", W);
	
	
	float* b = NULL;
	float* A = NULL;
	CPUMALLOC((void**)&b, sizeof(float) * LSH_M);
	CPUMALLOC((void**)&A, sizeof(float) * LSH_M * D);
	
	unsigned int* hashValues = NULL;
	GPUMALLOC((void**)&hashValues, sizeof(unsigned int) * nSamples);
	initMemory<unsigned int> <<< grid, threads>>>(hashValues, nSamples, 0);
	
	unsigned int* indices = NULL;
	GPUMALLOC((void**)&indices, sizeof(unsigned int) * nSamples);
	unsigned int* diff = NULL;
	GPUMALLOC((void**)&diff, sizeof(unsigned int) * nSamples);
	unsigned int* diff_sum = NULL;
	GPUMALLOC((void**)&diff_sum, sizeof(unsigned int) * nSamples);

	thrust::device_ptr<unsigned int> hashValues_ptr(hashValues);
	thrust::device_ptr<unsigned int> indices_ptr(indices);

	
	//for cuckoo hash table
	
	int c0, c1;
	int* c0s = NULL;
	int* c1s = NULL;
	int* constants = NULL;
	CPUMALLOC((void**)&c0s, sizeof(int) * CUCKOO_TABLE_NUM);
	CPUMALLOC((void**)&c1s, sizeof(int) * CUCKOO_TABLE_NUM);
	CPUMALLOC((void**)&constants, sizeof(int) * CUCKOO_TABLE_NUM * 2);

	unsigned int unUsedKey = 0; //the unUsed key (used in hash table)
	unsigned int* d_unUsedKey = NULL;
	GPUMALLOC((void**)&d_unUsedKey, sizeof(unsigned int));

	CUDPPConfiguration config;
	CUDPPHandle scanPlan;
	config.algorithm = CUDPP_SCAN;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_INT;
	cudppPlan(&scanPlan, config, nSamples, 1, 0);


	for(int i = 0; i < L; ++i)
	{
		for(int j = 0; j < LSH_M; ++j)
		{
			b[j] = rand() / (float)(RAND_MAX + 1) * W;
		}
		
		for(int j = 0; j < LSH_M * D; ++j)
		{
			A[j] = gaussian_rand(0, 1);
		}
		
		
		TOGPU_CONSTANT(constant_b, b, sizeof(float) * LSH_M, 0);
		TOGPU_CONSTANT(constant_A, A, sizeof(float) * D * LSH_M, 0);
		
		LSHHashComputation <<< grid, threads>>>(samples, hashValues, nSamples, dim, W);
		
		///////////////////////////////////////////////////////////////////////////////////
		///compute cuckoo hash table
		///////////////////////////////////////////////////////////////////////////////////
		
		// resort the samples according to hash(LSH)
		computePermutation <<< grid, threads>>>(nSamples, indices);

		cudaThreadSynchronize();

		//nvRadixSort::RadixSort rdsort(nSamples);
		//rdsort.sort(hashValues, indices, nSamples, 32);
		//hashingBasedElementGrouping(hashValues, indices, nSamples);
		thrust::stable_sort_by_key(hashValues_ptr, hashValues_ptr + nSamples, indices_ptr);
		
		gpu_diff <<< grid, threads>>>(hashValues, diff, nSamples);

		cudaThreadSynchronize();
		
		cudppScan(scanPlan, diff_sum, diff, nSamples);

		
		unsigned int nUniqueKeys = 0;
		FROMGPU(&nUniqueKeys, diff_sum + nSamples - 1, sizeof(unsigned int));
		printf("nUniqueKeys, %d\n", nUniqueKeys);
		
		unsigned int* uniqueKeys_starts = NULL;
		GPUMALLOC((void**)&uniqueKeys_starts, sizeof(unsigned int) * nUniqueKeys);
		unsigned int* uniqueKeys = NULL;
		GPUMALLOC((void**)&uniqueKeys, sizeof(unsigned int) * nUniqueKeys);
		unsigned int* uniqueKeys_counts = NULL;
		GPUMALLOC((void**)&uniqueKeys_counts, sizeof(unsigned int) * nUniqueKeys);
		unsigned int* uniqueKeys_indices = NULL;
		GPUMALLOC((void**)&uniqueKeys_indices, sizeof(unsigned int) * nUniqueKeys);
		
		
		dim3 uniqueKeysDim = makeGrid((int)ceilf(nUniqueKeys / (float)PROXIMITY_THREADS));
		dim3 uniqueKeysThreads = dim3(PROXIMITY_THREADS, 1, 1);
		
		computePermutation <<< uniqueKeysDim, uniqueKeysThreads>>>(nUniqueKeys, uniqueKeys_indices);
		gpu_compute_unique <<< grid, threads>>>(diff, hashValues, diff_sum, uniqueKeys_starts, uniqueKeys, nSamples);

		cudaThreadSynchronize();

		gpu_compute_unique_starts <<< uniqueKeysDim, uniqueKeysThreads>>>(uniqueKeys_starts, uniqueKeys_counts, nUniqueKeys, nSamples);

		gpu_find_unused_key <<< uniqueKeysDim, uniqueKeysThreads>>>(uniqueKeys, nUniqueKeys, d_unUsedKey);
		FROMGPU(&unUsedKey, d_unUsedKey, sizeof(unsigned int));
		
		
		// different LSH value corresponds to different LSH bucket.
		// the further hashing makes one hash(LSH) corresponds to several LSH buckets.
		// Now we store the hash(LSH) in the hash table
		//
		
		bool bIsConstructed = false;
		
		while(!bIsConstructed)
		{
			c0 = rand();
			c1 = rand();
			for(int k = 0; k < CUCKOO_TABLE_NUM * 2; ++k)
				constants[k] = rand();
				
			int ctmp = rand();
			for(int k = 0; k < CUCKOO_TABLE_NUM; ++k)
			{
				c0s[k] = ctmp ^ constants[2 * k];
				c1s[k] = ctmp ^ constants[2 * k + 1];
			}
			
			TOGPU_CONSTANT(constant_c0s, c0s, sizeof(int) * CUCKOO_TABLE_NUM, 0);
			TOGPU_CONSTANT(constant_c1s, c1s, sizeof(int) * CUCKOO_TABLE_NUM, 0);
			
			//printf("c0 c1: %d %d\n", c0, c1);
			//printf("unUsedKey: %d\n", unUsedKey);
			//for(int i = 0; i < CUCKOO_TABLE_NUM; ++i)
			//{
			//	printf("%d %d \n", c0s[i], c1s[i]);
			//}
			
			unsigned int* hashTable = NULL;
			unsigned int* hashTableValue = NULL;
			unsigned int hashTableSize;
			bIsConstructed = constructHashTable(uniqueKeys, uniqueKeys_indices, nUniqueKeys, unUsedKey, c0, c1,
			                                    hashTable, hashTableValue, hashTableSize);
			if(bIsConstructed)
			{
				LSHProximity <<< queryGrid, threads>>>(samples, queries, indices,
					uniqueKeys_starts, uniqueKeys_counts,
					hashTable, hashTableValue,
					proximityMaxHeaps, K,
					nUniqueKeys, nSamples, nQueries,
					dim, mindist,
					W,
					c0, c1, NULL);
			}
			
			GPUFREE(hashTable);
			GPUFREE(hashTableValue);
		}
		
		GPUFREE(uniqueKeys);
		GPUFREE(uniqueKeys_counts);
		GPUFREE(uniqueKeys_indices);
		GPUFREE(uniqueKeys_starts);
	}
	
	cudppDestroyPlan(scanPlan);
	
	float* outputDistances = NULL;
	GPUMALLOC((void**)&outputDistances, sizeof(float) * nQueries * K);
	
	gpu_hashing_postprocess <<< queryGrid, threads>>>(proximityMaxHeaps, nQueries, K, outputResult, outputDistances);
	
	unsigned int* h_proximityResults = NULL;
	float* h_proximityDistances = NULL;
	CPUMALLOC((void**)&h_proximityResults, sizeof(int) * nQueries * K);
	CPUMALLOC((void**)&h_proximityDistances, sizeof(float) * nQueries * K);
	FROMGPU(h_proximityDistances, outputDistances, sizeof(float) * nQueries * K);
	FROMGPU(h_proximityResults, outputResult, sizeof(unsigned int) * nQueries * K);
	FILE* resultfile = fopen("knnresult.txt", "w");
	FILE* distancefile = fopen("knndistance.txt", "w");
	for(int i = 0; i < nQueries; ++i)
	{
		for(int j = 0; j < K; ++j)
		{
			fprintf(resultfile, "%d ", h_proximityResults[j * nQueries + i]);
		}
		fprintf(resultfile, "\n");
		for(int j = 0; j < K; ++j)
		{
			fprintf(distancefile, "%f ", h_proximityDistances[j * nQueries + i]);
		}
		fprintf(distancefile, "\n");
	}
	fclose(resultfile);
	fclose(distancefile);
	CPUFREE(h_proximityResults);
	CPUFREE(h_proximityDistances);
	
	GPUFREE(outputDistances);
	GPUFREE(d_unUsedKey);
	GPUFREE(diff);
	GPUFREE(diff_sum);
	GPUFREE(hashValues);
	GPUFREE(indices);
	CPUFREE(b);
	CPUFREE(A);
	CPUFREE(c0s);
	CPUFREE(c1s);
	CPUFREE(constants);
	
	GPUFREE(proximityDistances);
	GPUFREE(proximityResults);
	GPUFREE(proximityMaxHeaps);

}


#endif