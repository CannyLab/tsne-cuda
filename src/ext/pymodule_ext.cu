
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
    pairwise_dist(handle, d_distances, d_points, N_POINTS, N_DIMS);

    // Copy the data back to the CPU
    thrust::copy(d_distances.begin(), d_distances.end(), dist);
}

void pymodule_naive_tsne(float *points, float *result, ssize_t *dims, int proj_dim, float learning_rate, float perplexity) {
    
    // Extract the dimensions of the points array
    ssize_t N_POINTS = dims[0];
    ssize_t N_DIMS = dims[1];

    // Construct device arrays
    thrust::device_vector<float> d_points(N_POINTS*N_DIMS);

    // Copy the points to the GPU using thrust
    thrust::copy(points, points+N_DIMS*N_POINTS, d_points.begin());

    // Construct the sigmas
    thrust::device_vector<float> sigmas(N_POINTS, 1.0f);

    /*
        RIGHT NOW, WE IGNORE THE PROJECTED DIMENSION, THE LEARNING RATE, and the PERPLEXITY
    */

    // Create the CUBLAS handle
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    // Do the T-SNE
    auto tsne_result = naive_tsne(handle, d_points, N_POINTS, N_DIMS);

    // Copy the data back to the CPU
    thrust::copy(tsne_result.begin(), tsne_result.end(), result);
}
