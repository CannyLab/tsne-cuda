/**
 * @brief Common includes for the Cuda-TSNE project
 *
 * @file common.h
 * @author David Chan
 * @date 2018-04-04
 */

 #ifndef COMMON_H
 #define COMMON_H

// CUDA Includes
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <cufft.h>

// Thrust includes
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

// C Library includes
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// C++ Library includes
#include <random>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>

#endif
