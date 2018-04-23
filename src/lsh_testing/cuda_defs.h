#ifndef __CUDA_DEFS_H_
#define __CUDA_DEFS_H_

#define GENERAL_THREADS 256

#define GPUMALLOC(D_POINTER, SIZE) CUDA_SAFE_CALL(cudaMalloc(D_POINTER, SIZE))
#define CPUMALLOC(H_POINTER, SIZE) CUDA_SAFE_CALL(cudaMallocHost(H_POINTER, SIZE))

//#define CPUFREE(H_POINTER)  if(H_POINTER) CUDA_SAFE_CALL(cudaFreeHost(H_POINTER))

#define CPUFREE(H_POINTER) \
	do { \
		if(H_POINTER) CUDA_SAFE_CALL(cudaFreeHost(H_POINTER)); \
		H_POINTER = NULL; \
	} while(0)

#define GPUFREE(D_POINTER) \
	do { \
		if(D_POINTER) CUDA_SAFE_CALL(cudaFree(D_POINTER)); \
		D_POINTER = NULL; \
	} while(0)

#define TOGPU(D_POINTER, H_POINTER, SIZE) CUDA_SAFE_CALL(cudaMemcpy(D_POINTER, H_POINTER, SIZE, cudaMemcpyHostToDevice))
#define FROMGPU(H_POINTER, D_POINTER, SIZE) CUDA_SAFE_CALL(cudaMemcpy(H_POINTER, D_POINTER, SIZE, cudaMemcpyDeviceToHost))
#define GPUTOGPU(D_TO, D_FROM, SIZE) CUDA_SAFE_CALL(cudaMemcpy(D_TO, D_FROM, SIZE, cudaMemcpyDeviceToDevice))

#define TOGPU_CONSTANT(D_POINTER, H_POINTER, SIZE, OFFSET) CUDA_SAFE_CALL(cudaMemcpyToSymbol(D_POINTER, H_POINTER, SIZE, OFFSET, cudaMemcpyHostToDevice))
#define GPUTOGPU_CONSTANT(D_TO, D_FROM, SIZE, OFFSET) CUDA_SAFE_CALL(cudaMemcpyToSymbol(D_TO, D_FORM, SIZE, OFFSET, cudaMemcpyDeviceToDevice))

#define GPUMEMSET(D_POINTER, INTVALUE, SIZE) CUDA_SAFE_CALL(cudaMemset(D_POINTER, INTVALUE, SIZE))


#endif