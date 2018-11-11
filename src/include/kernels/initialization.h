/**
 * @brief Kernels for computing t-SNE attractive forces with nearest neighbor approximation.
 *
 * @file apply_forces.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */
#ifndef SRC_INCLUDE_KERNELS_INITIALIZATION_H_
#define SRC_INCLUDE_KERNELS_INITIALIZATION_H_

#include "include/common.h"
#include "include/options.h"
#include "include/util/cuda_utils.h"

#include "include/kernels/apply_forces.h"
#include "include/kernels/bh_attr_forces.h"
#include "include/kernels/bh_rep_forces.h"
#include "include/kernels/bounding_box.h"
#include "include/kernels/perplexity_search.h"
#include "include/kernels/tree_builder.h"
#include "include/kernels/tree_sort.h"
#include "include/kernels/tree_summary.h"

//TSNE-Vars
extern __device__ volatile int stepd, bottomd, maxdepthd;
extern __device__ unsigned int blkcntd;
extern __device__ volatile float radiusd;


namespace tsnecuda {
namespace bh {
__global__ void InitializationKernel(int * __restrict errd);
void Initialize(tsnecuda::GpuOptions &gpu_opt, thrust::device_vector<int> &errd);
}
namespace naive {
void Initialize();
}
}

#endif
