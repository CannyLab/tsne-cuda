/**
 * @brief Barnes-Hut T-SNE implementation O(Nlog(N))
  * 
 * @file bh_tsne.h
 * @author David Chan
 * @date 2018-04-15
 */

#ifndef BH_TSNE_H
#define BH_TSNE_H

#include "common.h"
#include "util/cuda_utils.h"
#include "util/math_utils.h"
#include "util/matrix_broadcast_utils.h"
#include "util/reduce_utils.h"
#include "util/distance_utils.h"
#include "util/random_utils.h"
#include "util/thrust_utils.h"

// struct bounding_box_t {
//     float xmin;
//     float xmax;
//     float ymin;
//     float ymax;
// };

// namespace shit {
//     void compute_bounding_box(thrust::device_vector<float> &ys, const unsigned int N, float *xmin, float *xmax, float *ymin, float *ymax);

// thrust::device_vector<float> compute_pij(cublasHandle_t &handle, 
//                                          thrust::device_vector<float> &points, 
//                                          thrust::device_vector<float> &sigma, 
//                                          const unsigned int N, 
//                                          const unsigned int NDIMS);
// float compute_gradients(cublasHandle_t &handle, 
//                         thrust::device_vector<float> &forces,
//                         thrust::device_vector<float> &dist, 
//                         thrust::device_vector<float> &ys, 
//                         thrust::device_vector<float> &pij, 
//                         thrust::device_vector<float> &qij, 
//                         const unsigned int N,
//                         float eta);
// thrust::device_vector<float> bh_tsne(cublasHandle_t &handle, 
//                                      thrust::device_vector<float> &points, 
//                                      const unsigned int N, 
//                                      const unsigned int NDIMS,
//                                      const unsigned int PROJDIM);
    
// }


namespace BHTSNE {
    thrust::device_vector<float> tsne(cublasHandle_t &dense_handle, 
                                          cusparseHandle_t &sparse_handle,
                                          float* points, 
                                          unsigned int N_POINTS, 
                                          unsigned int N_DIMS, 
                                          unsigned int PROJDIM, 
                                          float perplexity, 
                                          float early_ex, 
                                          float learning_rate, 
                                          unsigned int n_iter, 
                                          unsigned int n_iter_np, 
                                          float min_g_norm,
                                          bool dump_points,
                                          bool interactive,
                                          float magnitude_factor,
                                          int init_type);
}

#endif
