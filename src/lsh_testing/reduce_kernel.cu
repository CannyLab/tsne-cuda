#include <cutil.h>
#include "cuda_utility.h"
#include "sharedmem.cuh"
/*
   This version adds multiple elements per thread sequentially.  This reduces the overall
   cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
   (Brent's Theorem optimization)
*/
template <class T, unsigned int blockSize>
__global__ void reduce_max_kernel(T* g_odata, T* g_idata, unsigned int n)
{
	SharedMemory<T> smem;
	volatile T* sdata = smem.getPointer();
	
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridSize).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	T thMax = fmaxf(g_idata[i], g_idata[i + blockSize]);
	
	i += gridSize;
	while(i < n)
	{
		T a = fmaxf(g_idata[i], g_idata[i + blockSize]);
		thMax = fmaxf(thMax, a);
		i += gridSize;
	}
	sdata[tid] = thMax;
	__syncthreads();
	
	// do reduction in shared mem
	if(blockSize >= 512)
	{
		if(tid < 256)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if(blockSize >= 256)
	{
		if(tid < 128)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if(blockSize >= 128)
	{
		if(tid <  64)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid +  64]);
		}
		__syncthreads();
	}
	
	if(tid < 32)
	{
		if(blockSize >=  64)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]);
		}
		if(blockSize >=  32)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]);
		}
		if(blockSize >=  16)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid +  8]);
		}
		if(blockSize >=   8)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid +  4]);
		}
		if(blockSize >=   4)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid +  2]);
		}
		if(blockSize >=   2)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid +  1]);
		}
	}
	
	// write result for this block to global mem
	if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
T reduce_max(T* d_idata, int size)
{
	const int maxThreads = 256;
	const int maxBlocks  = 64; // TODO: Test/Increase for future devices w/ more processors
	int threads = 1;
	
	if(size != 1)
	{
		threads = (size < maxThreads * 2) ? (size / 2) : maxThreads;
	}
	int blocks = size / (threads * 2);
	blocks = min(maxBlocks, blocks);
	
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

	T* d_odata = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, sizeof(T) * blocks));
	
	switch(threads)
	{
	case 512:
		reduce_max_kernel<T, 512> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 256:
		reduce_max_kernel<T, 256> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 128:
		reduce_max_kernel<T, 128> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 64:
		reduce_max_kernel<T,  64> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 32:
		reduce_max_kernel<T,  32> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 16:
		reduce_max_kernel<T,  16> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case  8:
		reduce_max_kernel<T,   8> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case  4:
		reduce_max_kernel<T,   4> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case  2:
		reduce_max_kernel<T,   2> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case  1:
		reduce_max_kernel<T,   1> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	default:
		exit(1);
	}
	
	T* h_odata = new T[blocks];
	CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, blocks * sizeof(T), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_odata));
	d_odata = NULL;
	
	T result = h_odata[0];
	for(int i = 1; i < blocks; i++)
	{
		result = max(result, h_odata[i]);
	}
	delete[] h_odata;
	
	return result;
}

// Instanciate kernels to prevent linker errors
template int    reduce_max<int   >(int    *d_idata, int size);
template float  reduce_max<float >(float  *d_idata, int size);
template double reduce_max<double>(double *d_idata, int size);



template <class T, unsigned int blockSize>
__global__ void reduce_min_kernel(T* g_odata, T* g_idata, unsigned int n)
{
	SharedMemory<T> smem;
	volatile T* sdata = smem.getPointer();
	
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridSize).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	T thMin = fminf(g_idata[i], g_idata[i + blockSize]);
	
	i += gridSize;
	while(i < n)
	{
		T a = fminf(g_idata[i], g_idata[i + blockSize]);
		thMin = fminf(thMin, a);
		i += gridSize;
	}
	sdata[tid] = thMin;
	__syncthreads();
	
	// do reduction in shared mem
	if(blockSize >= 512)
	{
		if(tid < 256)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid + 256]);
		}
		__syncthreads();
	}
	if(blockSize >= 256)
	{
		if(tid < 128)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid + 128]);
		}
		__syncthreads();
	}
	if(blockSize >= 128)
	{
		if(tid <  64)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid +  64]);
		}
		__syncthreads();
	}
	
	if(tid < 32)
	{
		if(blockSize >=  642)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid + 32]);
		}
		if(blockSize >=  32)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid + 16]);
		}
		if(blockSize >=  16)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid +  8]);
		}
		if(blockSize >= 8)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid +  4]);
		}
		if(blockSize >= 4)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid +  2]);
		}
		if(blockSize >=   2)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid +  1]);
		}
	}
	
	// write result for this block to global mem
	if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
T reduce_min(T* d_idata, int size)
{
	const int maxThreads = 256;
	const int maxBlocks  = 64; // TODO: Test/Increase for future devices w/ more processors
	int threads = 1;
	
	if(size != 1)
	{
		threads = (size < maxThreads * 2) ? size / 2 : maxThreads;
	}
	int blocks = size / (threads * 2);
	blocks = min(maxBlocks, blocks);
	
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	
	T* d_odata = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_odata, sizeof(T) * blocks));
	
	switch(threads)
	{
	case 512:
		reduce_min_kernel<T, 512> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 256:
		reduce_min_kernel<T, 256> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 128:
		reduce_min_kernel<T, 128> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 64:
		reduce_min_kernel<T,  64> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 32:
		reduce_min_kernel<T,  32> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case 16:
		reduce_min_kernel<T,  16> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case  8:
		reduce_min_kernel<T,   8> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case  4:
		reduce_min_kernel<T,   4> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case  2:
		reduce_min_kernel<T,   2> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	case  1:
		reduce_min_kernel<T,   1> <<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, size); break;
	default:
		exit(1);
	}
	
	T* h_odata = new T[blocks];
	CUDA_SAFE_CALL(cudaMemcpy(h_odata, d_odata, blocks * sizeof(T), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_odata));
	d_odata = NULL;
	
	T result = h_odata[0];
	for(int i = 1; i < blocks; i++)
	{
		result = min(result, h_odata[i]);
	}
	delete[] h_odata;
	
	return result;
}

// Instanciate kernels to prevent linker errors
template int    reduce_min<int   >(int    *d_idata, int size);
template float  reduce_min<float >(float  *d_idata, int size);
template double reduce_min<double>(double *d_idata, int size);


template <class T>
__global__ void get_elem_index_kernel(T* d_data, int size, T maxdata, int* d_maxidx)
{
	unsigned int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(threadId >= size) return;
	
	if(d_data[threadId] == maxdata)
	{
		//d_maxidx[0] = threadId;
		atomicExch(d_maxidx, threadId);
	}
}

template <class T> int get_elem_index(T* d_data, int size, T elem_data)
{
	dim3 dimGrid = makeGrid((int)ceilf(size / (float)256));
	dim3 dimBlock = dim3(256, 1, 1);
	
	
	int elem_idx;
	int* d_elem_idx = NULL;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_elem_idx, sizeof(int)));
	get_elem_index_kernel <<< dimGrid, dimBlock>>>(d_data, size, elem_data, d_elem_idx);
	CUDA_SAFE_CALL(cudaMemcpy(&elem_idx, d_elem_idx, sizeof(int), cudaMemcpyDeviceToHost));
	
	return elem_idx;
}

template int get_elem_index<int>(int* d_idata, int size, int maxdata);
template int get_elem_index<float>(float* d_idata, int size, float maxdata);
template int get_elem_index<double>(double* d_idata, int size, double maxdata);
