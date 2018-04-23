#ifndef __CUDA_HASH_H_
#define __CUDA_HASH_H_

#include "defs.h"
#include "cuda_utility.h"
#include "cuda_defs.h"
#include <cudpp/cudpp.h>
#include <cutil.h>
#include "cuda_timer.h"

__device__ __constant__ int constant_c0s[CUCKOO_TABLE_NUM];
__device__ __constant__ int constant_c1s[CUCKOO_TABLE_NUM];


__device__ unsigned int findBucketPerKey(int key, int c0, int c1, unsigned int nBuckets)
{
	unsigned int tmp1 = c0 + c1 * key;
	unsigned int tmp2 = tmp1 / 1900813;
	tmp1 = tmp1 - tmp2 * 1900813;
	tmp2 = tmp1 / nBuckets;
	tmp1 = tmp1 - tmp2 * nBuckets;
	return tmp1;
}

// deprecated: replaced by findBucket 
__global__ void findBucket(unsigned int* keys, unsigned int* counts, unsigned int* offsets, unsigned int* buckets, unsigned int nKeys, unsigned int nBuckets, int c0, int c1)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nKeys)
		return;
		
	int key = keys[threadId];
	
	unsigned int bucket = findBucketPerKey(key, c0, c1, nBuckets);
	
	buckets[threadId] = bucket;
	
	int offset = atomicAdd(counts + bucket, 1);
	offsets[threadId] = offset;
}

// compute bucket id for each key
// in: keys, out: buckets
__global__ void findBucket_phase1(unsigned int* keys, unsigned int* buckets, unsigned int nKeys, unsigned int nBuckets, int c0, int c1)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nKeys)
		return;

	int key = keys[threadId];

	unsigned int bucket = findBucketPerKey(key, c0, c1, nBuckets);

	buckets[threadId] = bucket;
}

// compute start for non-empty buckets and the id of next non-empty bucket
// in: buckets, out: starts, follows,
__global__ void findBucket_phase2(unsigned int* buckets, unsigned int* starts, unsigned int* stops, unsigned int nKeys)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nKeys)
		return;

	if(threadId == 0)
	{
		starts[buckets[0]] = 0;
	}
	else if(buckets[threadId] != buckets[threadId-1]) 
	{
		starts[buckets[threadId]] = threadId;	
		stops[buckets[threadId - 1]] = threadId;
	}
}

// compute count number for each bucket
// in: starts, follows, out: counts
__global__ void findBucket_phase3(unsigned int* starts, unsigned int* stops, unsigned int* counts, unsigned int nKeys, unsigned int nBuckets)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nBuckets)
		return;

	if(threadId == nBuckets - 1)
	{
		counts[threadId] = nKeys - starts[threadId];
	}
	else
	{
		if(stops[threadId] != -1)
			counts[threadId] = stops[threadId] - starts[threadId];
	}
}

// compute local offsets for each key in the corresponding bucket
// in: starts, indices, buckets, out: offsets
__global__ void findBucket_phase4(unsigned int* starts, unsigned int* buckets, unsigned int* indices, unsigned int* offsets, unsigned int nKeys)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nKeys)
		return;

	offsets[indices[threadId]] = threadId - starts[buckets[threadId]];
}

// buckets order is changed
void findBucket(unsigned int* keys, unsigned int* counts, unsigned int* offsets, unsigned int* starts, unsigned int* buckets, unsigned int* indices, unsigned int nKeys, unsigned int nBuckets, int c0, int c1)
{
	dim3 grid = makeGrid((int)ceilf(nKeys / (float)CUCKOO_HASH_THREADS));
	dim3 gridBuckets = makeGrid((int)ceilf(nBuckets / (float)CUCKOO_HASH_THREADS));
	dim3 threads = dim3(CUCKOO_HASH_THREADS, 1, 1);
	findBucket_phase1 <<< grid, threads >>> (keys, buckets, nKeys, nBuckets, c0, c1);

	unsigned int* stops = NULL;
	GPUMALLOC((void**)&stops, sizeof(unsigned int) * nBuckets);
	initMemory <unsigned int> <<< grid, gridBuckets >>> (stops, nBuckets, -1);

	cudaThreadSynchronize();

	unsigned int nCompareBits = ceilf(log((float)nBuckets) / log(2.0f));

	nvRadixSort::RadixSort rdsort(nKeys);
	rdsort.sort(buckets, indices, nKeys, nCompareBits);

	cudaThreadSynchronize();

	findBucket_phase2 <<< grid, threads >>> (buckets, starts, stops, nKeys);

	cudaThreadSynchronize();

	findBucket_phase3 <<< gridBuckets, threads >>> (starts, stops, counts, nKeys, nBuckets);
	findBucket_phase4 <<< grid, threads >>> (starts, buckets, indices, offsets, nKeys);

	GPUFREE(stops);
}

template <typename T>
__global__ void rearrangeKeysAndValues(unsigned int* newKeys, T* newValues, unsigned int nKeys, unsigned int nBuckets, unsigned int* keys, T* values, unsigned int* starts, unsigned int* offsets, unsigned int* buckets)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nKeys)
		return;
		
	unsigned int bucket = buckets[threadId];
	unsigned int start = starts[bucket];
	unsigned int offset = offsets[threadId];
	
	newKeys[start + offset] = keys[threadId];
	newValues[start + offset] = values[threadId];
}

__global__ void rearrangeKeys(unsigned int* newKeys, unsigned int nKeys, unsigned int nBuckets, unsigned int* keys, unsigned int* starts, unsigned int* offsets, unsigned int* buckets)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nKeys)
		return;

	unsigned int bucket = buckets[threadId];
	unsigned int start = starts[bucket];
	unsigned int offset = offsets[threadId];

	newKeys[start + offset] = keys[threadId];
}


template <typename T>
__global__ void rearrangeKeysAndValues(unsigned int* newKeys, T* newValues, unsigned int* keys, T* values, unsigned int* indices, unsigned int nKeys)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nKeys)
		return;

	newKeys[threadId] = keys[indices[threadId]];
	newValues[threadId] = values[indices[threadId]];
}

__global__ void rearrangeKeys(unsigned int* newKeys, unsigned int* keys, unsigned int* indices, unsigned int nKeys)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= nKeys)
		return;

	newKeys[threadId] = keys[indices[threadId]];
}

__device__ unsigned int compute_cuckoo_hash_value(unsigned int key, int c0, int c1)
{
	unsigned int tmp1 = (unsigned int)c0 + (unsigned int)c1 * key;
	unsigned int tmp2 = tmp1 / 1900813;
	tmp1 = tmp1 - tmp2 * 1900813;
	tmp2 = tmp1 / CUCKOO_HASH_PER_BLOCK_SIZE;
	tmp1 = tmp1 - tmp2 * CUCKOO_HASH_PER_BLOCK_SIZE;
	return tmp1;
}

template <typename T>
__global__ void cuckoo_hashing(unsigned int* keys, T* values, unsigned int* starts, unsigned int* counts, bool* placed, unsigned int nKeys, unsigned int nBuckets, unsigned int unUsedKey, int* pIsSuceed, unsigned int* globalHashTable, T* globalHashTableValue)
{
	unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	unsigned int threadId = threadIdx.x;
	
	if(blockId >= nBuckets)
		return;
		
	__shared__ unsigned int hashTable[CUCKOO_MULTIHASH_PER_BLOCK_SIZE]; // 192 * 3
	__shared__ T hashTableValue[CUCKOO_MULTIHASH_PER_BLOCK_SIZE];
	
	__shared__ unsigned int init_iter;
	if(threadId == 0)
		init_iter = (int)ceilf(CUCKOO_MULTIHASH_PER_BLOCK_SIZE / (float)CUCKOO_BLOCK_SIZE);
		
	__syncthreads();
	
	for(int i = 0; i < init_iter; ++i)
	{
		int offset = i * CUCKOO_BLOCK_SIZE + threadId;
		if(offset < CUCKOO_MULTIHASH_PER_BLOCK_SIZE)
			hashTable[offset] = unUsedKey;
	}
	
	__syncthreads();
	
	__shared__ unsigned int count; //count per block
	__shared__ unsigned int iter; //iter per block
	__shared__ unsigned int start; //start per block
	
	__shared__ int bSuccess; //whether a block cuckoo succeed
	__shared__ unsigned int nUnplaced; //the number of unplaced key in a block
	
	if(threadId == 0)
	{
		count = counts[blockId];
		iter = (int)ceilf(count / (float)CUCKOO_BLOCK_SIZE);
		start = starts[blockId];
		bSuccess = 0;
	}
	
	__syncthreads();
	
	
	
	for(int at = 0; at < CUCKOO_MAX_ATTEMPT; ++at)
	{
		for(int i = 0; i < CUCKOO_TABLE_NUM; ++i)
		{
			for(int j = 0; j < iter; ++j)
			{
				unsigned int offset = j * CUCKOO_BLOCK_SIZE + threadId; //local offset
				if(offset < count)
				{
					if(!placed[start + offset])
					{
						unsigned int key = keys[start + offset];
						unsigned int hash_value = compute_cuckoo_hash_value(key,  constant_c0s[i], constant_c1s[i]);
						unsigned int* hash_address = hashTable + i * CUCKOO_HASH_PER_BLOCK_SIZE + hash_value;
						atomicExch(hash_address, key);
					}
				}
				
				__syncthreads();
			}
			
			if(threadId == 0)
				nUnplaced = 0;
			__syncthreads();
			
			for(int j = 0; j < iter; ++j)
			{
				unsigned int offset = j * CUCKOO_BLOCK_SIZE + threadId; //local offset;
				
				if(offset < count)
				{
					bool placed_ = false;
					unsigned int key = keys[start + offset];
					for(int k = 0; k < CUCKOO_TABLE_NUM; ++k)
					{
						unsigned int hash_value = compute_cuckoo_hash_value(key, constant_c0s[k], constant_c1s[k]);
						unsigned int* hash_address = hashTable + k * CUCKOO_HASH_PER_BLOCK_SIZE + hash_value;
						
						if(*hash_address == key)
						{
							T* hash_value_address = hashTableValue + k * CUCKOO_HASH_PER_BLOCK_SIZE + hash_value;
							*hash_value_address = values[start + offset];
							placed_ = true;
							break;
						}
					}
					
					if(placed_)
						placed[start + offset] = true;
					else
					{
						atomicAdd(&nUnplaced, 1);
						placed[start + offset] = false;
					}
					
				}
				__syncthreads();
			}
			
			
			if(threadId == 0)
			{
				if(nUnplaced == 0)
					bSuccess = true;
			}
			
			__syncthreads();
			
			if(bSuccess == 1)
				break;
		}
		
		if(bSuccess == 1)
			break;
	}
	__syncthreads();
	
	if(threadId == 0)
	{
		atomicExch(pIsSuceed, bSuccess);
	}
	
	__syncthreads();
	
	if(bSuccess == 1) // copy table back
	{
		for(int i = 0; i < init_iter; ++i)
		{
			unsigned int offset = i * CUCKOO_BLOCK_SIZE + threadId;
			if(offset < CUCKOO_MULTIHASH_PER_BLOCK_SIZE)
			{
				int local_i = offset / CUCKOO_HASH_PER_BLOCK_SIZE;
				int local_offset = offset - local_i * CUCKOO_HASH_PER_BLOCK_SIZE;
				int global_offset = local_i * nBuckets * CUCKOO_HASH_PER_BLOCK_SIZE + blockId * CUCKOO_HASH_PER_BLOCK_SIZE + local_offset;
				globalHashTable[global_offset] = hashTable[offset];
				globalHashTableValue[global_offset] = hashTableValue[offset];
			}
		}
	}
}


__global__ void cuckoo_hashing(unsigned int* keys, unsigned int* starts, unsigned int* counts, bool* placed, unsigned int nKeys, unsigned int nBuckets, unsigned int unUsedKey, int* pIsSuceed, unsigned int* globalHashTable)
{
	unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	unsigned int threadId = threadIdx.x;

	if(blockId >= nBuckets)
		return;

	__shared__ unsigned int hashTable[CUCKOO_MULTIHASH_PER_BLOCK_SIZE]; // 192 * 3

	__shared__ unsigned int init_iter;
	if(threadId == 0)
		init_iter = (int)ceilf(CUCKOO_MULTIHASH_PER_BLOCK_SIZE / (float)CUCKOO_BLOCK_SIZE);

	__syncthreads();

	for(int i = 0; i < init_iter; ++i)
	{
		int offset = i * CUCKOO_BLOCK_SIZE + threadId;
		if(offset < CUCKOO_MULTIHASH_PER_BLOCK_SIZE)
			hashTable[offset] = unUsedKey;
	}

	__syncthreads();

	__shared__ unsigned int count; //count per block
	__shared__ unsigned int iter; //iter per block
	__shared__ unsigned int start; //start per block

	__shared__ int bSuccess; //whether a block cuckoo succeed
	__shared__ unsigned int nUnplaced; //the number of unplaced key in a block

	if(threadId == 0)
	{
		count = counts[blockId];
		iter = (int)ceilf(count / (float)CUCKOO_BLOCK_SIZE);
		start = starts[blockId];
		bSuccess = 0;
	}

	__syncthreads();



	for(int at = 0; at < CUCKOO_MAX_ATTEMPT; ++at)
	{
		for(int i = 0; i < CUCKOO_TABLE_NUM; ++i)
		{
			for(int j = 0; j < iter; ++j)
			{
				unsigned int offset = j * CUCKOO_BLOCK_SIZE + threadId; //local offset
				if(offset < count)
				{
					if(!placed[start + offset])
					{
						unsigned int key = keys[start + offset];
						unsigned int hash_value = compute_cuckoo_hash_value(key,  constant_c0s[i], constant_c1s[i]);
						unsigned int* hash_address = hashTable + i * CUCKOO_HASH_PER_BLOCK_SIZE + hash_value;
						atomicExch(hash_address, key);
					}
				}

				__syncthreads();
			}

			if(threadId == 0)
				nUnplaced = 0;
			__syncthreads();

			for(int j = 0; j < iter; ++j)
			{
				unsigned int offset = j * CUCKOO_BLOCK_SIZE + threadId; //local offset;

				if(offset < count)
				{
					bool placed_ = false;
					unsigned int key = keys[start + offset];
					for(int k = 0; k < CUCKOO_TABLE_NUM; ++k)
					{
						unsigned int hash_value = compute_cuckoo_hash_value(key, constant_c0s[k], constant_c1s[k]);
						unsigned int* hash_address = hashTable + k * CUCKOO_HASH_PER_BLOCK_SIZE + hash_value;

						if(*hash_address == key)
						{
							placed_ = true;
							break;
						}
					}

					if(placed_)
						placed[start + offset] = true;
					else
					{
						atomicAdd(&nUnplaced, 1);
						placed[start + offset] = false;
					}

				}
				__syncthreads();
			}


			if(threadId == 0)
			{
				if(nUnplaced == 0)
					bSuccess = true;
			}

			__syncthreads();

			if(bSuccess == 1)
				break;
		}

		if(bSuccess == 1)
			break;
	}
	__syncthreads();

	if(threadId == 0)
	{
		atomicExch(pIsSuceed, bSuccess);
	}

	__syncthreads();

	if(bSuccess == 1) // copy table back
	{
		for(int i = 0; i < init_iter; ++i)
		{
			unsigned int offset = i * CUCKOO_BLOCK_SIZE + threadId;
			if(offset < CUCKOO_MULTIHASH_PER_BLOCK_SIZE)
			{
				int local_i = offset / CUCKOO_HASH_PER_BLOCK_SIZE;
				int local_offset = offset - local_i * CUCKOO_HASH_PER_BLOCK_SIZE;
				globalHashTable[local_i * nBuckets * CUCKOO_HASH_PER_BLOCK_SIZE + blockId * CUCKOO_HASH_PER_BLOCK_SIZE + local_offset] = hashTable[offset];
			}
		}
	}
}

//hashTable and hashTableValue are allocated in the function, but deallocated in the hashProximityComputation function.
template <typename T>
bool constructHashTable(unsigned int* keys, T* values, unsigned int nKeys, unsigned int unUsedKey,
                        int c0, int c1, unsigned int*& hashTable, T*& hashTableValue, unsigned int& hashTableSize)
{
	unsigned int nBuckets = (unsigned int)ceil((float)nKeys / CUCKOO_HASH_PARAMETER);
	dim3 grid = makeGrid((int)ceilf(nKeys / (float)CUCKOO_HASH_THREADS));
	dim3 threads = dim3(CUCKOO_HASH_THREADS, 1, 1);
	
	unsigned int* counts = NULL;
	GPUMALLOC((void**)&counts, sizeof(unsigned int) * nBuckets);
	GPUMEMSET(counts, 0, sizeof(unsigned int) * nBuckets);
	
	unsigned int* offsets = NULL;
	GPUMALLOC((void**)&offsets, sizeof(unsigned int) * nKeys);
	GPUMEMSET(offsets, 0, sizeof(unsigned int) * nKeys);
	
	unsigned int* buckets = NULL;
	GPUMALLOC((void**)&buckets, sizeof(unsigned int) * nKeys);
	GPUMEMSET(buckets, 0, sizeof(unsigned int) * nKeys);
	
	unsigned int* starts = NULL;
	GPUMALLOC((void**)&starts, sizeof(unsigned int) * nBuckets);
	GPUMEMSET(starts, 0, sizeof(unsigned int) * nBuckets);

	unsigned int* indices = NULL;
	GPUMALLOC((void**)&indices, sizeof(unsigned int) * nKeys);
	computePermutation <<< grid, threads >>>(nKeys, indices);

	unsigned int* rearrangedKeys = NULL;
	T* rearrangedValues = NULL;
	GPUMALLOC((void**)&rearrangedKeys, sizeof(unsigned int) * nKeys);
	GPUMALLOC((void**)&rearrangedValues, sizeof(T) * nKeys);

	findBucket(keys, counts,offsets, starts, buckets, indices, nKeys, nBuckets, c0, c1);

	cudaThreadSynchronize();

	rearrangeKeysAndValues<T> <<< grid, threads>>>(rearrangedKeys, rearrangedValues, keys, values, indices, nKeys);

	cudaThreadSynchronize();

	GPUFREE(indices);

	unsigned int cuckoo_table_size = nBuckets * CUCKOO_MULTIHASH_PER_BLOCK_SIZE / CUCKOO_TABLE_NUM;
	
	
	GPUMALLOC((void**)&hashTable, sizeof(unsigned int) * CUCKOO_TABLE_NUM * cuckoo_table_size);
	GPUMALLOC((void**)&hashTableValue, sizeof(T) * CUCKOO_TABLE_NUM * cuckoo_table_size);
	hashTableSize = CUCKOO_TABLE_NUM * cuckoo_table_size;
	
	dim3 HTdim = makeGrid((int)ceilf(CUCKOO_TABLE_NUM * cuckoo_table_size / (float)CUCKOO_HASH_THREADS));
	dim3 HTthreads = dim3(CUCKOO_HASH_THREADS, 1, 1);
	initMemory<unsigned int> <<< HTdim, HTthreads>>>(hashTable, CUCKOO_TABLE_NUM * cuckoo_table_size, unUsedKey);
	//initMemory<unsigned int> <<< HTdim, HTthreads>>>(hashTable, CUCKOO_TABLE_NUM * cuckoo_table_size, -1);
	
	bool* placed = NULL;
	GPUMALLOC((void**)&placed, sizeof(bool) * nKeys);
	initMemory<bool> <<< grid, threads>>>(placed, nKeys, false);
	
	dim3 CUCKOOdim = makeGrid(nBuckets);
	dim3 CUCKOOthreads = dim3(CUCKOO_BLOCK_SIZE, 1, 1);
	
	int bSuccess = 0;
	int* pIsSucceed = NULL;
	GPUMALLOC((void**)&pIsSucceed, sizeof(int));
	
	cuckoo_hashing<T> <<< CUCKOOdim, CUCKOOthreads>>>(rearrangedKeys, rearrangedValues, starts, counts, placed, nKeys, nBuckets, unUsedKey, pIsSucceed, hashTable, hashTableValue);
	
	FROMGPU(&bSuccess, pIsSucceed, sizeof(int));
	cudaThreadSynchronize();
	
	GPUFREE(counts);
	GPUFREE(offsets);
	GPUFREE(buckets);
	GPUFREE(starts);
	GPUFREE(rearrangedKeys);
	GPUFREE(rearrangedValues);
	GPUFREE(placed);
	GPUFREE(pIsSucceed);

	if(bSuccess == 1) return true;
	else return false;
	
}



////hashTable and hashTableValue are allocated in the function, but deallocated in the hashProximityComputation function.
//template <typename T>
//bool constructHashTable(unsigned int* keys, T* values, unsigned int nKeys, unsigned int unUsedKey,
//						int c0, int c1, unsigned int*& hashTable, T*& hashTableValue, unsigned int& hashTableSize)
//{
//	unsigned int nBuckets = (unsigned int)ceil((float)nKeys / CUCKOO_HASH_PARAMETER);
//	dim3 grid = makeGrid((int)ceilf(nKeys / (float)CUCKOO_HASH_THREADS));
//	dim3 threads = dim3(CUCKOO_HASH_THREADS, 1, 1);
//
//	unsigned int* counts = NULL;
//	GPUMALLOC((void**)&counts, sizeof(unsigned int) * nBuckets);
//	GPUMEMSET(counts, 0, sizeof(unsigned int) * nBuckets);
//
//	unsigned int* offsets = NULL;
//	GPUMALLOC((void**)&offsets, sizeof(unsigned int) * nKeys);
//	GPUMEMSET(offsets, 0, sizeof(unsigned int) * nKeys);
//
//	unsigned int* buckets = NULL;
//	GPUMALLOC((void**)&buckets, sizeof(unsigned int) * nKeys);
//	GPUMEMSET(buckets, 0, sizeof(unsigned int) * nKeys);
//
//	unsigned int* starts = NULL;
//	GPUMALLOC((void**)&starts, sizeof(unsigned int) * nBuckets);
//	GPUMEMSET(starts, 0, sizeof(unsigned int) * nBuckets);
//
//	unsigned int* rearrangedKeys = NULL;
//	T* rearrangedValues = NULL;
//	GPUMALLOC((void**)&rearrangedKeys, sizeof(unsigned int) * nKeys);
//	GPUMALLOC((void**)&rearrangedValues, sizeof(T) * nKeys);
//
//	findBucket <<< grid, threads>>>(keys, counts, offsets, buckets, nKeys, nBuckets, c0, c1);
//
//	cudaThreadSynchronize();
//
//	CUDPPConfiguration config;
//	CUDPPHandle scanPlan;
//	config.algorithm = CUDPP_SCAN;
//	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
//	config.op = CUDPP_ADD;
//	config.datatype = CUDPP_INT;
//	cudppPlan(&scanPlan, config, nKeys, 1, 0);
//	cudppScan(scanPlan, starts, counts, nBuckets);
//	cudppDestroyPlan(scanPlan);
//
//	rearrangeKeysAndValues<T> <<< grid, threads>>>(rearrangedKeys, rearrangedValues, nKeys, nBuckets, keys, values, starts, offsets, buckets);
//
//	cudaThreadSynchronize();
//
//	unsigned int cuckoo_table_size = nBuckets * CUCKOO_MULTIHASH_PER_BLOCK_SIZE / CUCKOO_TABLE_NUM;
//
//
//	GPUMALLOC((void**)&hashTable, sizeof(unsigned int) * CUCKOO_TABLE_NUM * cuckoo_table_size);
//	GPUMALLOC((void**)&hashTableValue, sizeof(T) * CUCKOO_TABLE_NUM * cuckoo_table_size);
//	hashTableSize = CUCKOO_TABLE_NUM * cuckoo_table_size;
//
//	dim3 HTdim = makeGrid((int)ceilf(CUCKOO_TABLE_NUM * cuckoo_table_size / (float)CUCKOO_HASH_THREADS));
//	dim3 HTthreads = dim3(CUCKOO_HASH_THREADS, 1, 1);
//	initMemory<unsigned int> <<< HTdim, HTthreads>>>(hashTable, CUCKOO_TABLE_NUM * cuckoo_table_size, unUsedKey);
//	//initMemory<unsigned int> <<< HTdim, HTthreads>>>(hashTable, CUCKOO_TABLE_NUM * cuckoo_table_size, -1);
//
//	bool* placed = NULL;
//	GPUMALLOC((void**)&placed, sizeof(bool) * nKeys);
//	initMemory<bool> <<< grid, threads>>>(placed, nKeys, false);
//
//	dim3 CUCKOOdim = makeGrid(nBuckets);
//	dim3 CUCKOOthreads = dim3(CUCKOO_BLOCK_SIZE, 1, 1);
//
//	int bSuccess = 0;
//	int* pIsSucceed = NULL;
//	GPUMALLOC((void**)&pIsSucceed, sizeof(int));
//
//	cuckoo_hashing<T> <<< CUCKOOdim, CUCKOOthreads>>>(rearrangedKeys, rearrangedValues, starts, counts, placed, nKeys, nBuckets, unUsedKey, pIsSucceed, hashTable, hashTableValue);
//
//	FROMGPU(&bSuccess, pIsSucceed, sizeof(int));
//	cudaThreadSynchronize();
//
//	GPUFREE(counts);
//	GPUFREE(offsets);
//	GPUFREE(buckets);
//	GPUFREE(starts);
//	GPUFREE(rearrangedKeys);
//	GPUFREE(rearrangedValues);
//	GPUFREE(placed);
//	GPUFREE(pIsSucceed);
//
//	if(bSuccess == 1) return true;
//	else return false;
//
//}

//bool constructHashTable(unsigned int* keys, unsigned int nKeys, unsigned int unUsedKey,
//						int c0, int c1, unsigned int*& hashTable, unsigned int& hashTableSize)
//{
//
//	unsigned int nBuckets = (unsigned int)ceil((float)nKeys / CUCKOO_HASH_PARAMETER);
//	dim3 grid = makeGrid((int)ceilf(nKeys / (float)CUCKOO_HASH_THREADS));
//	dim3 threads = dim3(CUCKOO_HASH_THREADS, 1, 1);
//
//	unsigned int* counts = NULL;
//	GPUMALLOC((void**)&counts, sizeof(unsigned int) * nBuckets);
//	GPUMEMSET(counts, 0, sizeof(unsigned int) * nBuckets);
//
//	unsigned int* offsets = NULL;
//	GPUMALLOC((void**)&offsets, sizeof(unsigned int) * nKeys);
//	GPUMEMSET(offsets, 0, sizeof(unsigned int) * nKeys);
//
//	unsigned int* buckets = NULL;
//	GPUMALLOC((void**)&buckets, sizeof(unsigned int) * nKeys);
//	GPUMEMSET(buckets, 0, sizeof(unsigned int) * nKeys);
//
//	unsigned int* starts = NULL;
//	GPUMALLOC((void**)&starts, sizeof(unsigned int) * nBuckets);
//	GPUMEMSET(starts, 0, sizeof(unsigned int) * nBuckets);
//
//	unsigned int* indices = NULL;
//	GPUMALLOC((void**)&indices, sizeof(unsigned int) * nKeys);
//	computePermutation <<< grid, threads >>>(nKeys, indices);
//
//	unsigned int* rearrangedKeys = NULL;
//	GPUMALLOC((void**)&rearrangedKeys, sizeof(unsigned int) * nKeys);
//
//	findBucket(keys, counts,offsets, starts, buckets, indices, nKeys, nBuckets, c0, c1);
//
//	cudaThreadSynchronize();
//
//	rearrangeKeys <<< grid, threads>>>(rearrangedKeys, keys, indices, nKeys);
//
//	cudaThreadSynchronize();
//
//	GPUFREE(indices);
//
//	unsigned int cuckoo_table_size = nBuckets * CUCKOO_MULTIHASH_PER_BLOCK_SIZE / CUCKOO_TABLE_NUM;
//
//	GPUMALLOC((void**)&hashTable, sizeof(unsigned int) * CUCKOO_TABLE_NUM * cuckoo_table_size);
//	hashTableSize = CUCKOO_TABLE_NUM * cuckoo_table_size;
//
//	dim3 HTdim = makeGrid((int)ceilf(CUCKOO_TABLE_NUM * cuckoo_table_size / (float)CUCKOO_HASH_THREADS));
//	dim3 HTthreads = dim3(CUCKOO_HASH_THREADS, 1, 1);
//	initMemory<unsigned int> <<< HTdim, HTthreads>>>(hashTable, CUCKOO_TABLE_NUM * cuckoo_table_size, unUsedKey);
//
//	bool* placed = NULL;
//	GPUMALLOC((void**)&placed, sizeof(bool) * nKeys);
//	initMemory<bool> <<< grid, threads>>>(placed, nKeys, false);
//
//	dim3 CUCKOOdim = makeGrid(nBuckets);
//	dim3 CUCKOOthreads = dim3(CUCKOO_BLOCK_SIZE, 1, 1);
//
//	int bSuccess = 0;
//	int* pIsSucceed = NULL;
//	GPUMALLOC((void**)&pIsSucceed, sizeof(int));
//
//	cudaThreadSynchronize();
//
//	cuckoo_hashing <<< CUCKOOdim, CUCKOOthreads>>>(rearrangedKeys, starts, counts, placed, nKeys, nBuckets, unUsedKey, pIsSucceed, hashTable);
//
//
//	FROMGPU(&bSuccess, pIsSucceed, sizeof(int));
//
//	GPUFREE(counts);
//	GPUFREE(offsets);
//	GPUFREE(buckets);
//	GPUFREE(starts);
//	GPUFREE(rearrangedKeys);
//	GPUFREE(placed);
//	GPUFREE(pIsSucceed);
//
//	if(bSuccess == 1) return true;
//	else return false;
//
//}

bool constructHashTable(unsigned int* keys, unsigned int nKeys, unsigned int unUsedKey,
						int c0, int c1, unsigned int*& hashTable, unsigned int& hashTableSize)
{

	unsigned int nBuckets = (unsigned int)ceil((float)nKeys / CUCKOO_HASH_PARAMETER);

	unsigned int* counts = NULL;
	GPUMALLOC((void**)&counts, sizeof(unsigned int) * nBuckets);
	GPUMEMSET(counts, 0, sizeof(unsigned int) * nBuckets);

	unsigned int* offsets = NULL;
	GPUMALLOC((void**)&offsets, sizeof(unsigned int) * nKeys);
	GPUMEMSET(offsets, 0, sizeof(unsigned int) * nKeys);

	unsigned int* buckets = NULL;
	GPUMALLOC((void**)&buckets, sizeof(unsigned int) * nKeys);
	GPUMEMSET(buckets, 0, sizeof(unsigned int) * nKeys);

	unsigned int* starts = NULL;
	GPUMALLOC((void**)&starts, sizeof(unsigned int) * nBuckets);
	GPUMEMSET(starts, 0, sizeof(unsigned int) * nBuckets);

	dim3 grid = makeGrid((int)ceilf(nKeys / (float)CUCKOO_HASH_THREADS));
	dim3 threads = dim3(CUCKOO_HASH_THREADS, 1, 1);
	findBucket <<< grid, threads>>>(keys, counts, offsets, buckets, nKeys, nBuckets, c0, c1);
	cudaThreadSynchronize();


	CUDPPConfiguration config;
	CUDPPHandle scanPlan;
	config.algorithm = CUDPP_SCAN;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_INT;
	cudppPlan(&scanPlan, config, nKeys, 1, 0);
	cudppScan(scanPlan, starts, counts, nBuckets);
	cudppDestroyPlan(scanPlan);

	unsigned int* rearrangedKeys = NULL;
	GPUMALLOC((void**)&rearrangedKeys, sizeof(unsigned int) * nKeys);


	rearrangeKeys <<< grid, threads>>>(rearrangedKeys, nKeys, nBuckets, keys, starts, offsets, buckets);

	unsigned int cuckoo_table_size = nBuckets * CUCKOO_MULTIHASH_PER_BLOCK_SIZE / CUCKOO_TABLE_NUM;

	GPUMALLOC((void**)&hashTable, sizeof(unsigned int) * CUCKOO_TABLE_NUM * cuckoo_table_size);
	hashTableSize = CUCKOO_TABLE_NUM * cuckoo_table_size;

	dim3 HTdim = makeGrid((int)ceilf(CUCKOO_TABLE_NUM * cuckoo_table_size / (float)CUCKOO_HASH_THREADS));
	dim3 HTthreads = dim3(CUCKOO_HASH_THREADS, 1, 1);
	initMemory<unsigned int> <<< HTdim, HTthreads>>>(hashTable, CUCKOO_TABLE_NUM * cuckoo_table_size, unUsedKey);

	bool* placed = NULL;
	GPUMALLOC((void**)&placed, sizeof(bool) * nKeys);
	initMemory<bool> <<< grid, threads>>>(placed, nKeys, false);

	dim3 CUCKOOdim = makeGrid(nBuckets);
	dim3 CUCKOOthreads = dim3(CUCKOO_BLOCK_SIZE, 1, 1);

	int bSuccess = 0;
	int* pIsSucceed = NULL;
	GPUMALLOC((void**)&pIsSucceed, sizeof(int));

	cudaThreadSynchronize();

	cuckoo_hashing <<< CUCKOOdim, CUCKOOthreads>>>(rearrangedKeys, starts, counts, placed, nKeys, nBuckets, unUsedKey, pIsSucceed, hashTable);


	FROMGPU(&bSuccess, pIsSucceed, sizeof(int));

	GPUFREE(counts);
	GPUFREE(offsets);
	GPUFREE(buckets);
	GPUFREE(starts);
	GPUFREE(rearrangedKeys);
	GPUFREE(placed);
	GPUFREE(pIsSucceed);

	if(bSuccess == 1) return true;
	else return false;

}

#endif