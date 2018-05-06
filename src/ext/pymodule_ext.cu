
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
    tsnecuda::util::PairwiseDistance(handle, d_distances, d_points, N_POINTS, N_DIMS);

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

    // Construct the options
    BHTSNE::Options opt(result, points, N_POINTS, N_DIMS);
    opt.perplexity = perplexity;
    opt.learning_rate = learning_rate;
    opt.early_exaggeration = early_ex;
    opt.iterations = n_iter;
    opt.iterations_no_progress = n_iter_np;
    opt.n_neighbors = 32;
    opt.min_gradient_norm = min_g_norm;
    
    // Return data setup
    opt.return_style = BHTSNE::RETURN_STYLE::ONCE;

    // Do the t-SNE
    BHTSNE::tsne(dense_handle, sparse_handle, opt);

    // Copy the data back from the GPU
    cudaDeviceSynchronize();
}

void pymodule_bhsnapshot(float *points, float *result, ssize_t *dims, int proj_dim, float perplexity, float early_ex, 
    float learning_rate, int n_iter,  int n_iter_np, float min_g_norm, float* preinit_data, int num_snapshots) {

    // Extract the dimensions of the points array
    ssize_t N_POINTS = dims[0];
    ssize_t N_DIMS = dims[1];

    // Create the CUBLAS handles
    cublasHandle_t dense_handle;
    cublasSafeCall(cublasCreate(&dense_handle));
    cusparseHandle_t sparse_handle;
    cusparseSafeCall(cusparseCreate(&sparse_handle));

    // Construct the options
    BHTSNE::Options opt(result, points, N_POINTS, N_DIMS);
    opt.perplexity = perplexity;
    opt.learning_rate = learning_rate;
    opt.early_exaggeration = early_ex;
    opt.iterations = n_iter;
    opt.iterations_no_progress = n_iter_np;
    opt.n_neighbors = 32;
    opt.min_gradient_norm = min_g_norm;
    
    // Fancy initialization
    opt.preinit_data = preinit_data;
    opt.initialization = BHTSNE::TSNE_INIT::VECTOR;
    
    // Return data setup
    opt.return_style = BHTSNE::RETURN_STYLE::SNAPSHOT;
    opt.num_snapshots = num_snapshots;

    // Do the t-SNE
    BHTSNE::tsne(dense_handle, sparse_handle, opt);

    // Copy the data back from the GPU
    cudaDeviceSynchronize();
}