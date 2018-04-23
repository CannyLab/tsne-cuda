#ifndef __REDUCTION_KERNEL_H__
#define __REDUCTION_KERNEL_H__

template <class T> T reduce_max(T* d_idata, int size);

template <class T> T reduce_min(T* d_idata, int size);

template <class T> int get_elem_index(T* d_data, int size, T elem_data);

#endif // #ifndef __REDUCTION_KERNEL_H__