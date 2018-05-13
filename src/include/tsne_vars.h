/**
 * @brief Barnes-hut t-SNE global variables
 *
 * @file apply_forces.cu
 * @author Roshan Rao
 * @date 2018-05-08
 * Copyright (c) 2018, Regents of the University of California
 */
#ifndef INITIALIZATION_VARS
#ifndef SRC_INCLUDE_KERNELS_TSNEVARS_H_
#define SRC_INCLUDE_KERNELS_TSNEVARS_H_

extern __device__ volatile int stepd, bottomd, maxdepthd;
extern __device__ unsigned int blkcntd;
extern __device__ volatile float radiusd;

#endif
#endif


