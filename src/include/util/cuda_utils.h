#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include "common.h"

__host__ __device__ int iDivUp(int, int);
extern "C" void cusolveSafeCall(cusolverStatus_t);
extern "C" void cublasSafeCall(cublasStatus_t);
extern "C" void cufftSafeCall(cufftResult err);
extern "C" void cusparseSafeCall(cusparseStatus_t err);
extern "C" void gpuErrchk(cudaError_t ans);

template <class T>
void reverseArray(const T * __restrict__, T * __restrict__, const int, const T a = static_cast<T>(1));

template<class T>
T h_l2_norm(T *v1, T *v2, const int N);

template <class T>
void linearCombination(const T * __restrict__, const T * __restrict__, T * __restrict__, const int, const int, const cublasHandle_t);

void linearCombination(const float * __restrict__, const float * __restrict__, float * __restrict__,
	const int, const int, const cublasHandle_t);

void linearCombination(const double * __restrict__, const double * __restrict__, double * __restrict__,
	const int, const int, const cublasHandle_t);

template<class T>
void vectorAddConstant(T * __restrict__, const T, const int);

template<class T>
void vectorMulConstant(T * __restrict__, const T, const int);

template<class T>
void h_vectorMulConstant(T * __restrict__, const T, const int);

template<class T>
__host__ __device__ T fma2(T, T, T);

__device__ int modulo(int, int);

__device__ double atomicAdd(double *, double);
__device__ float  atomicMin(float *, float);

double deg2rad(double);

void cudaMemoryUsage();
	
/**************************/
/* TEMPLATE SHARED MEMORY */
/**************************/
// --- Credit to the simpleTemplates CUDA sample
template <typename T>
struct SharedMemory
{
	// Ensure that we won't compile any un-specialized types
	__device__ T *getPointer()
	{
		extern __device__ void error(void);
		error();
		return NULL;
	}
};

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

template <>
struct SharedMemory <int>
{
	__device__ int *getPointer()
	{
		extern __shared__ int s_int[];
		return s_int;
	}
};

template <>
struct SharedMemory <unsigned int>
{
	__device__ unsigned int *getPointer()
	{
		extern __shared__ unsigned int s_uint[];
		return s_uint;
	}
};

template <>
struct SharedMemory <char>
{
	__device__ char *getPointer()
	{
		extern __shared__ char s_char[];
		return s_char;
	}
};

template <>
struct SharedMemory <unsigned char>
{
	__device__ unsigned char *getPointer()
	{
		extern __shared__ unsigned char s_uchar[];
		return s_uchar;
	}
};

template <>
struct SharedMemory <short>
{
	__device__ short *getPointer()
	{
		extern __shared__ short s_short[];
		return s_short;
	}
};

template <>
struct SharedMemory <unsigned short>
{
	__device__ unsigned short *getPointer()
	{
		extern __shared__ unsigned short s_ushort[];
		return s_ushort;
	}
};

template <>
struct SharedMemory <long>
{
	__device__ long *getPointer()
	{
		extern __shared__ long s_long[];
		return s_long;
	}
};

template <>
struct SharedMemory <unsigned long>
{
	__device__ unsigned long *getPointer()
	{
		extern __shared__ unsigned long s_ulong[];
		return s_ulong;
	}
};

template <>
struct SharedMemory <bool>
{
	__device__ bool *getPointer()
	{
		extern __shared__ bool s_bool[];
		return s_bool;
	}
};

template <>
struct SharedMemory <float>
{
	__device__ float *getPointer()
	{
		extern __shared__ float s_float[];
		return s_float;
	}
};

template <>
struct SharedMemory <double>
{
	__device__ double *getPointer()
	{
		extern __shared__ double s_double[];
		return s_double;
	}
};

void printarray(thrust::device_vector<float> vec, const unsigned int N, const unsigned int M);

#endif
