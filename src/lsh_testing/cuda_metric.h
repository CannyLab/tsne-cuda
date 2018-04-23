#ifndef __CUDA_METRIC_H_
#define __CUDA_METRIC_H_

template <typename T>
__device__ __inline__ float distance_sqr2(T* a, T* b, unsigned int dim)
{
	float t = 0;
	for(unsigned int i = 0; i < dim; ++i)
	{
		t += (a[i] - b[i]) * (a[i] - b[i]);
	}
	
	return t;
}

template <typename T>
__device__ __inline__ float distance_sqr2_interleaved(T* a, T* b, unsigned int span_a, unsigned int span_b, unsigned int dim)
{
	float t = 0, tmp = 0;
	for(unsigned int i = 0; i < dim; ++i)
	{
		tmp = (a[i * span_a] - b[i * span_b]);
		t += tmp * tmp;
	}

	return t;
}


#endif