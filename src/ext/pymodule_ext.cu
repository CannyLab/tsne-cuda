
// Implementation file for the python extensions

#include "ext/pymodule_ext.h"

void pymodule_e_dist(float *points, float *dist, ssize_t *dims) {
    
    // Extract the dimensions of the points array
    ssize_t N_POINTS = dims[0];
    ssize_t N_DIMS = dims[1];

    // Construct device arrays
    thrust::device_vector<float> d_points(N_POINTS*N_DIMS);
    thrust::device_vector<float> d_distances(N_POINTS*N_POINTS);

    // Copy the points to the GPU using thrust
    thrust::copy(points, points+N_DIMS*N_POINTS, d_points.begin());

    // Create the handle, and construct the points
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));
    Distance::pairwise_dist(handle, d_distances, d_points, N_POINTS, N_DIMS);

    // Copy the data back to the CPU
    thrust::copy(d_distances.begin(), d_distances.end(), dist);
}

void pymodule_naive_tsne(float *points, float *result, ssize_t *dims, int proj_dim, float perplexity, float early_ex, 
                            float learning_rate, int n_iter,  int n_iter_np, float min_g_norm) {
    
    // Extract the dimensions of the points array
    ssize_t N_POINTS = dims[0];
    ssize_t N_DIMS = dims[1];

    // Construct device arrays
    thrust::device_vector<float> d_points(N_POINTS*N_DIMS);

    // Copy the points to the GPU using thrust
    thrust::copy(points, points+N_DIMS*N_POINTS, d_points.begin());

    // Construct the sigmas
    thrust::device_vector<float> sigmas(N_POINTS, 1.0f);

    // Create the CUBLAS handle
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    // Do the T-SNE
    auto tsne_result = NaiveTSNE::tsne(handle, d_points, N_POINTS, N_DIMS, proj_dim, perplexity, 
                                            early_ex, learning_rate, n_iter, n_iter_np, min_g_norm);

    // Copy the data back to the CPU
    thrust::copy(tsne_result.begin(), tsne_result.end(), result);
}

void pymodule_compute_pij(float *points, float* sigmas, float *result, ssize_t *dims) {

     // Extract the dimensions of the points array
     ssize_t N_POINTS = dims[0];
     ssize_t N_DIMS = dims[1];
 
     // Construct device arrays
     thrust::device_vector<float> d_points(N_POINTS*N_DIMS);
 
     // Copy the points to the GPU using thrust
     thrust::copy(points, points+N_DIMS*N_POINTS, d_points.begin());
 
     // Construct the sigmas
     thrust::device_vector<float> d_sigmas(N_POINTS);
     thrust::copy(sigmas, sigmas+N_POINTS, d_sigmas.begin());
 
     // Create the CUBLAS handle
     cublasHandle_t handle;
     cublasSafeCall(cublasCreate(&handle));
 
     // Do the T-SNE
     thrust::device_vector<float> pij(N_POINTS*N_POINTS);
     NaiveTSNE::compute_pij(handle, pij, d_points, d_sigmas, N_POINTS, N_DIMS);
 
     // Copy the data back to the CPU
     thrust::copy(pij.begin(), pij.end(), result);

}

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
      float min_g_norm);

void pymodule_bh_tsne(float *points, float *result, ssize_t *dims, int proj_dim, float perplexity, float early_ex, 
    float learning_rate, int n_iter,  int n_iter_np, float min_g_norm) {

    // Extract the dimensions of the points array
    ssize_t N_POINTS = dims[0];
    ssize_t N_DIMS = dims[1];

    // Create the CUBLAS handles
    cublasHandle_t dense_handle;
    cublasSafeCall(cublasCreate(&dense_handle));
    cusparseHandle_t sparse_handle;
    cusparseSafeCall(cusparseCreate(&sparse_handle));

    // Do the t-SNE
    thrust::device_vector<float> tsne_results = BHTSNE::tsne(dense_handle, sparse_handle, points, 
                                                              N_POINTS, N_DIMS, 2, perplexity, early_ex, learning_rate, 
                                                              n_iter, n_iter_np, min_g_norm, false, false, 5.0, 0, 1023, "tcp://localhost:5556", nullptr);

    // Copy the data back from the GPU
    thrust::copy(tsne_results.begin(), tsne_results.end(), result);
}