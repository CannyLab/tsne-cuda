/**
 * @brief Assorted thrust vector utilities.
 * 
 * @file thrust_utils.h
 * @author your name
 * @date 2018-05-05
 * Copyright (c) 2018, Regents of the University of California
 */


#ifndef SRC_INCLUDE_UTIL_THRUST_UTILS_H_
#define SRC_INCLUDE_UTIL_THRUST_UTILS_H_

#include <functional>

#include "include/common.h"

namespace tsnecuda {
namespace util {

// Strided iterator from
// https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
template <typename Iterator>
class StridedRange {
 public:
    typedef typename thrust::iterator_difference<Iterator>::type DifferenceType;
    struct StrideFunctor :
        public thrust::unary_function<DifferenceType, DifferenceType> {
        DifferenceType stride;
        explicit StrideFunctor(DifferenceType stride)
            : stride(stride) {}

        __host__ __device__ DifferenceType operator()(
                const DifferenceType & x) const {
            return stride * x;
        }
    };

    typedef typename thrust::counting_iterator<DifferenceType> CountingIterator;
    typedef typename thrust::transform_iterator<StrideFunctor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator, TransformIterator> PermutationIterator;

    typedef PermutationIterator iterator;

    StridedRange(Iterator first, Iterator last, DifferenceType stride) :
            first(first), last(last), stride(stride) {}

    iterator begin(void) const {
        return PermutationIterator(first,
            TransformIterator(CountingIterator(0), StrideFunctor(stride)));
    }

    iterator end(void) const {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

 protected:
    Iterator first;
    Iterator last;
    DifferenceType stride;
};

/**
 * @brief Zero the diagonal of the matrix
 * 
 * @param d_vector 
 */
void ZeroDeviceMatrixDiagonal(thrust::device_vector<float> &d_vector,
                              const int N);

}  // namespace util
}  // namespace tsnecuda


#endif  // SRC_INCLUDE_UTIL_THRUST_UTILS_H_
