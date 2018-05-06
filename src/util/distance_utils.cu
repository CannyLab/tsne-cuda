/**
 * @brief Implementation of different distances
 * 
 * @file distance_utils.cu
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#include "include/util/distance_utils.h"

struct func_sqrt {
    __host__ __device__ float operator()(const float &x) const {
            return pow(x, 0.5); }
};

// This really does a simultaneous row/col matrix vector broadcast
// to compute ||x^2|| + ||y^2|| - 2 x^Ty.
// Added fabs to deal with numerical instabilities. I think this is a
// reasonable solution
__global__ void tsne::util::AssembleDistances(
        const float * __restrict__ d_squared_norms,
        float * __restrict__ d_dot_products,
        const uint32_t num_points) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i < num_points) && (j < num_points))
        d_dot_products[i * num_points + j] = fabs(d_squared_norms[j] +
            d_squared_norms[i] - 2 * d_dot_products[i * num_points + j]);
    }

// Code from https://github.com/OrangeOwlSolutions/cuBLAS/blob/master/All_pairs_distances.cu
// Expects num_points x num_dims matrix in points
// Squared norms taken from diagnoal of dot product which should be faster
// and result in actually zeroing out the diagonal in assemble_final_result

void tsne::util::SquaredPairwiseDistance(cublasHandle_t &handle,
        thrust::device_vector<float> &d_distances,
        const thrust::device_vector<float> &d_points,
        const uint32_t num_points,
        const uint32_t num_dims) {
    const uint32_t kBlockSize = 16;
    float kAlpha = 1.f;
    float kBeta = 0.f;

    // TODO(Roshan): Could replace this with cublasSsyrk, might be faster?
    cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, num_points,
        num_points, num_dims, &kAlpha,
        thrust::raw_pointer_cast(d_points.data()),
        num_points, thrust::raw_pointer_cast(d_points.data()),
        num_points, &kBeta,
        thrust::raw_pointer_cast(d_distances.data()), num_points));

    typedef thrust::device_vector<float>::iterator Iterator;
    strided_range<Iterator> diagonalized(d_distances.begin(),
            d_distances.end(), num_points + 1);
    thrust::device_vector<float> squared_norms(num_points);
    thrust::copy(diagonalized.begin(), diagonalized.end(),
                squared_norms.begin());

    dim3 kBlockDimensions(kBlockSize, kBlockSize);
    dim3 kGridDimensions(iDivUp(num_points, kBlockSize),
            iDivUp(num_points, kBlockSize));
    tsne::util::AssembleDistances<<<kGridDimensions, kBlockDimensions>>>(
        thrust::raw_pointer_cast(squared_norms.data()),
        thrust::raw_pointer_cast(d_distances.data()), num_points);
}

void tsne::util::PairwiseDistance(cublasHandle_t &handle,
        thrust::device_vector<float> &d_distances,
        const thrust::device_vector<float> &d_points,
        const uint32_t num_points,
        const uint32_t num_dims) {
    tsne::util::SquaredPairwiseDistance(handle, d_distances, d_points,
                                          num_points, num_dims);
    tsne::util::SqrtDeviceVector(d_distances, d_distances);
}

void tsne::util::KNearestNeighbors(int64_t* indices, float* distances,
        const float* const points, const uint32_t num_dims,
        const uint32_t num_points, const uint32_t num_near_neighbors) {
    const int32_t kNumCells = static_cast<int32_t>(
            std::sqrt(static_cast<float>(num_points)));
    const int32_t kNumCellsToProbe = 20;

    if (num_near_neighbors < 1024) {
        // Construct the GPU resources necessary
        faiss::gpu::StandardGpuResources faiss_resources;
        // res.noTempMemory();

        // Construct the GPU configuration object
        faiss::gpu::GpuIndexIVFFlatConfig faiss_config;

        // TODO(David): Allow for dynamic device placement
        faiss_config.device = 0;

        faiss_config.indicesOptions = faiss::gpu::INDICES_32_BIT;
        faiss_config.flatConfig.useFloat16 = false;
        faiss_config.useFloat16IVFStorage = false;

        faiss::gpu::GpuIndexIVFFlat search_index(&faiss_resources,
                num_dims, kNumCells, faiss::METRIC_L2, faiss_config);
        search_index.setNumProbes(kNumCellsToProbe);
        search_index.train(num_points, points);
        search_index.add(num_points, points);

        // Perform the KNN query
        search_index.search(num_points, points, num_near_neighbors,
                            distances, indices);
    } else {
        // Construct the index table on the CPU (since the GPU
        // can only handle 1023 neighbors)
        faiss::IndexFlatL2 quantizer(num_dims);
        faiss::IndexIVFFlat search_index(&quantizer, num_dims, kNumCells,
                                  faiss::METRIC_L2);
        search_index.train(num_points, points);
        search_index.add(num_points, points);

        // Perform the KNN query
        search_index.nprobe = kNumCellsToProbe;
        search_index.search(num_points, points, num_near_neighbors,
                     distances, indices);
    }
}
