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

#include "include/kernels/apply_forces.h"
#include "include/kernels/bh_attr_forces.h"
#include "include/kernels/bh_rep_forces.h"
#include "include/kernels/bounding_box.h"
#include "include/kernels/perplexity_search.h"
#include "include/kernels/tree_builder.h"
#include "include/kernels/tree_sort.h"
#include "include/kernels/tree_summary.h"


namespace tsnecuda {
namespace bh {
__global__ void InitializationKernel(int * __restrict errd)
{
    *errd = 0;
    stepd = -1;
    maxdepthd = 1;
    blkcntd = 0;
}

void Initialize(int * __restrict__ errd) 
}

namespace naive {

void Initialize();

}
}

#endif