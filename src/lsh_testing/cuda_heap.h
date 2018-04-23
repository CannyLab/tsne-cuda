#ifndef __CUDA_HEAP_H_
#define __CUDA_HEAP_H_


struct CUDA_MinHeap
{
	int size;
	void *list;
	void *list_assoc;
};

template <class T, class U>
__device__ void insert(CUDA_MinHeap *minheap, T c, U c_assoc)
{
	int i = minheap->size;
	minheap->size++;
	T *list = (T*)minheap->list;
	U *list_assoc = (U*)minheap->list_assoc;
	while(i > 0 && list[(i - 1) / 2] > c)
	{
		list[i] = list[(i - 1) / 2];
		list_assoc[i] = list_assoc[(i - 1) / 2];
		i = (i - 1) / 2;
	}
	list[i] = c;
	list_assoc[i] = c_assoc;
}

template <class T, class U>
__device__ void min_heapify(CUDA_MinHeap *minheap, unsigned int i)
{
	int l = (i << 1) + 1;
	int r = l + 1;
	T *list = (T*)minheap->list;
	U *list_assoc = (U*)minheap->list_assoc;
	while(l < minheap->size)
	{
		unsigned int smallest = i;
		if(list[l] < list[i])
			smallest = l;
		if(r < minheap->size && list[r] < list[smallest])
			smallest = r;
			
		if(smallest != i)
		{
			float c = list[i];
			unsigned int c_assoc = list_assoc[i];
			list[i] = list[smallest];
			list_assoc[i] = list_assoc[smallest];
			list[smallest] = c;
			list_assoc[smallest] = c_assoc;
			i = smallest;
		}
		else
			break;
			
		l = (i << 1) + 1;
		r = l + 1;
	}
}

template <class T, class U>
__device__ bool extract_min(CUDA_MinHeap *minheap, T *c, U *c_assoc)
{
	if(minheap->size >= 1)
	{
		T *list = (T*)minheap->list;
		U *list_assoc = (U*)minheap->list_assoc;
		*c = list[0];
		*c_assoc = list_assoc[0];
		
		list[0] = list[minheap->size - 1];
		list_assoc[0] = list_assoc[minheap->size - 1];
		minheap->size--;
		min_heapify<T, U>(minheap, 0);
		return true;
	}
	else
		return false;
}




struct CUDA_MaxHeap
{
	unsigned int size;
	void *list;
	void *list_assoc;
};

template <class T, class U>
__device__ __inline__ void insert(CUDA_MaxHeap *maxheap, T c, U c_assoc)
{
	int i = maxheap->size;
	maxheap->size++;
	T* list = (T*)maxheap->list;
	U* list_assoc = (U*)maxheap->list_assoc;
	int parent = (i - 1) / 2;
	while(i > 0 && list[parent] < c)
	{
		list[i] = list[parent];
		list_assoc[i] = list_assoc[parent];
		i = parent;
		parent = (i - 1) / 2;
	}
	list[i] = c;
	list_assoc[i] = c_assoc;
}

template <class T, class U>
__device__ __inline__ void max_heapify(CUDA_MaxHeap *maxheap, unsigned int i)
{
	int l = (i << 1) + 1;
	int r = l + 1;
	T* list = (T*)maxheap->list;
	U* list_assoc = (U*)maxheap->list_assoc;
	while(l < maxheap->size)
	{
		unsigned int largest = i;
		if(list[l] > list[i])
			largest = l;
		if(r < maxheap->size && list[r] > list[largest])
			largest = r;
			
		if(largest != i)
		{
			float c = list[i];
			unsigned int c_assoc = list_assoc[i];
			list[i] = list[largest];
			list_assoc[i] = list_assoc[largest];
			list[largest] = c;
			list_assoc[largest] = c_assoc;
			
			i = largest;
		}
		else
			break;
			
		l = (i << 1) + 1;
		r = l + 1;
	}
}

template <class T, class U>
__device__ __inline__ bool extract_max(CUDA_MaxHeap *maxheap, T *c, U *c_assoc)
{
	if(maxheap->size >= 1)
	{
		T* list = (T*)maxheap->list;
		U* list_assoc = (U*)maxheap->list_assoc;
		*c = list[0];
		*c_assoc = list_assoc[0];
		
		list[0] = list[maxheap->size - 1];
		list_assoc[0] = list_assoc[maxheap->size - 1];
		maxheap->size--;
		max_heapify<T, U>(maxheap, 0);
		return true;
	}
	else
		return false;
}


struct CUDA_MaxHeap_Interleaved
{
	unsigned int size;
	void *list;
	void *list_assoc;
};

template <class T, class U>
__device__ __inline__ void insert(CUDA_MaxHeap_Interleaved *maxheap, T c, U c_assoc, unsigned int maxsize)
{
	int i = maxheap->size;
	maxheap->size++;
	T* list = (T*)maxheap->list;
	U* list_assoc = (U*)maxheap->list_assoc;
	int parent = (i - 1) / 2;
	while(i > 0 && list[parent * maxsize] < c)
	{
		list[i * maxsize] = list[parent * maxsize];
		list_assoc[i * maxsize] = list_assoc[parent * maxsize];
		i = parent;
		parent = (i - 1) / 2;
	}
	list[i * maxsize] = c;
	list_assoc[i * maxsize] = c_assoc;
}

template <class T, class U>
__device__ __inline__ void max_heapify(CUDA_MaxHeap_Interleaved *maxheap, unsigned int i, unsigned int maxsize)
{
	int l = (i << 1) + 1;
	int r = l + 1;
	T* list = (T*)maxheap->list;
	U* list_assoc = (U*)maxheap->list_assoc;
	while(l < maxheap->size)
	{
		unsigned int largest = i;
		if(list[l * maxsize] > list[i* maxsize])
			largest = l;
		if(r < maxheap->size && list[r* maxsize] > list[largest* maxsize])
			largest = r;
			
		if(largest != i)
		{
			float c = list[i * maxsize];
			unsigned int c_assoc = list_assoc[i * maxsize];
			list[i * maxsize] = list[largest * maxsize];
			list_assoc[i * maxsize] = list_assoc[largest * maxsize];
			list[largest * maxsize] = c;
			list_assoc[largest * maxsize] = c_assoc;
			
			i = largest;
		}
		else
			break;
			
		l = (i << 1) + 1;
		r = l + 1;
	}
}

template <class T, class U>
__device__ __inline__ bool extract_max(CUDA_MaxHeap_Interleaved *maxheap, T *c, U *c_assoc, unsigned int maxsize)
{
	if(maxheap->size >= 1)
	{
		T* list = (T*)maxheap->list;
		U* list_assoc = (U*)maxheap->list_assoc;
		*c = list[0];
		*c_assoc = list_assoc[0];
		
		list[0] = list[(maxheap->size - 1) * maxsize];
		list_assoc[0] = list_assoc[(maxheap->size - 1) * maxsize];
		maxheap->size--;
		max_heapify<T, U>(maxheap, 0, maxsize);
		return true;
	}
	else
		return false;
}




#endif