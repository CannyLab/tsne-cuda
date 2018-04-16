/**
 * @brief Assorted Thrust utilities
 * 
 * @file test_tsne.h
 * @date 2018-04-11
 */


#ifndef THRUST_UTILS_H
#define THRUST_UTILS_H

#include "common.h"

// Strided iterator from https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
template <typename Iterator>
class strided_range
{
    public:
        typedef typename thrust::iterator_difference<Iterator>::type difference_type;

        struct stride_functor : public thrust::unary_function<difference_type,difference_type>
        {
            difference_type stride;
            stride_functor(difference_type stride)
                : stride(stride) {}

            __host__ __device__
            difference_type operator()(const difference_type & i) const
            {
                return stride * i;
            }
        };

        typedef typename thrust::counting_iterator<difference_type> CountingIterator;
        typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
        typedef typename thrust::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

        typedef PermutationIterator iterator;

        strided_range(Iterator first, Iterator last, difference_type stride)
            : first(first), last(last), stride(stride) {}

        iterator begin(void) const
        {
            return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
        }

        iterator end(void) const
        {
            return begin() + ((last - first) + (stride - 1)) / stride;
        }

    protected:
        Iterator first;
        Iterator last;
        difference_type stride;
};

template <typename T>
struct pair_t
{
    T first;
    T second;
};

template <typename T>
struct minmax_unary_op : public thrust::unary_function< T, pair_t<T> >
{
    __host__ __device__ pair_t<T> operator()(const T& x) const {
        pair_t<T> result;
        result.first = x;
        result.second = x;
        return result;
    }
};

template <typename T>
struct minmax_binary_op : public thrust::binary_function< pair_t<T>, pair_t<T>, pair_t<T> >
{
  __host__ __device__ pair_t<T> operator()(const pair_t<T>& x, const pair_t<T>& y) const {
        pair_t<T> result;
        result.first = thrust::min(x.first, y.first);
        result.first = thrust::max(x.first, y.first);
        return result;
    }
};

void zero_diagonal(thrust::device_vector<float> &vec, const unsigned int N);



#endif
