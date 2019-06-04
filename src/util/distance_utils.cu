/**
 * @brief Implementation of different distances
 *
 * @file distance_utils.cu
 * @author David Chan
 * @date 2018-04-04
 * Copyright (c) 2018, Regents of the University of California
 */

#include "include/util/distance_utils.h"
#include <chrono>
#include <future>

// This really does a simultaneous row/col matrix vector broadcast
// to compute ||x^2|| + ||y^2|| - 2 x^Ty.
// Added fabs to deal with numerical instabilities. I think this is a
// reasonable solution
__global__ void tsnecuda::util::AssembleDistances(
        const float * __restrict__ d_squared_norms,
        float * __restrict__ d_dot_products,
        const int num_points) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i < num_points) && (j < num_points))
        d_dot_products[i * num_points + j] = fabs(d_squared_norms[j] +
            d_squared_norms[i] - 2 * d_dot_products[i * num_points + j]);
    }

// Code from https://github.com/OrangeOwlSolutions/cuBLAS/blob/master/All_pairs_distances.cu
// Expects num_points x num_dims matrix in points
// Squared norms taken from diagnoal of dot product which should be faster
// and result in actually zeroing out the diagonal in assemble_final_result
void tsnecuda::util::SquaredPairwiseDistance(cublasHandle_t &handle,
        thrust::device_vector<float> &d_distances,
        const thrust::device_vector<float> &d_points,
        const int num_points,
        const int num_dims) {
    const int kBlockSize = 16;
    float kAlpha = 1.f;
    float kBeta = 0.f;

    // TODO(Roshan): Could replace this with cublasSsyrk, might be faster?
    CublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, num_points,
        num_points, num_dims, &kAlpha,
        thrust::raw_pointer_cast(d_points.data()),
        num_points, thrust::raw_pointer_cast(d_points.data()),
        num_points, &kBeta,
        thrust::raw_pointer_cast(d_distances.data()), num_points));

    typedef thrust::device_vector<float>::iterator Iterator;
    tsnecuda::util::StridedRange<Iterator> diagonalized(d_distances.begin(),
            d_distances.end(), num_points + 1);
    thrust::device_vector<float> squared_norms(num_points);
    thrust::copy(diagonalized.begin(), diagonalized.end(),
                squared_norms.begin());

    dim3 kBlockDimensions(kBlockSize, kBlockSize);
    dim3 kGridDimensions(iDivUp(num_points, kBlockSize),
            iDivUp(num_points, kBlockSize));
    tsnecuda::util::AssembleDistances<<<kGridDimensions, kBlockDimensions>>>(
        thrust::raw_pointer_cast(squared_norms.data()),
        thrust::raw_pointer_cast(d_distances.data()), num_points);
}

void tsnecuda::util::PairwiseDistance(cublasHandle_t &handle,
        thrust::device_vector<float> &d_distances,
        const thrust::device_vector<float> &d_points,
        const int num_points,
        const int num_dims) {
    tsnecuda::util::SquaredPairwiseDistance(handle, d_distances, d_points,
                                          num_points, num_dims);
    tsnecuda::util::SqrtDeviceVector(d_distances, d_distances);
}

void tsnecuda::util::KNearestNeighbors(tsnecuda::GpuOptions &gpu_opt,
        int64_t* indices, float* distances,
        const float* const points, const int num_dims,
        const int num_points, const int num_near_neighbors) {
    const int32_t kNumCells = static_cast<int32_t>(
            std::sqrt(static_cast<float>(num_points)));
    const int32_t kNumCellsToProbe = 20;

    if (true) {
        // const int32_t kSubQuant = 2;
        // const int32_t kBPC = 8;
        faiss::gpu::StandardGpuResources faiss_resources;
        // faiss::gpu::StandardGpuResources faiss_resources_2;
        // faiss_resources.noTempMemory();
        // faiss_resources_2.noTempMemory();

        // Construct the GPU configuration object
        // faiss::gpu::GpuIndexIVFPQConfig faiss_config;
        // faiss::gpu::GpuIndexIVFPQConfig faiss_config_2;
        faiss::gpu::GpuIndexIVFFlatConfig faiss_config;
        // faiss::gpu::GpuIndexIVFFlatConfig faiss_config_2;


        // // TODO(David): Allow for dynamic device placement
        faiss_config.device = gpu_opt.device;
        // faiss_config_2.device = 1;

        faiss_config.indicesOptions = faiss::gpu::INDICES_32_BIT;
        faiss_config.flatConfig.useFloat16 = false;
        faiss_config.useFloat16IVFStorage = false;

        // faiss_config_2.indicesOptions = faiss::gpu::INDICES_32_BIT;
        // faiss_config_2.flatConfig.useFloat16 = false;
        // faiss_config_2.useFloat16IVFStorage = false;


        // faiss_config.indicesOptions = faiss::gpu::INDICES_32_BIT;
        // faiss_config.useFloat16LookupTables = true;
        // faiss_config.usePrecomputedTables = true;

        // faiss_config_2.indicesOptions = faiss::gpu::INDICES_32_BIT;
        // faiss_config_2.useFloat16LookupTables = true;
        // faiss_config_2.usePrecomputedTables = true;

        // faiss_config.useFloat16IVFStorage = false;
        faiss::gpu::GpuIndexIVFFlat search_index(&faiss_resources, num_dims, kNumCells, faiss::METRIC_L2, faiss_config);
        // faiss::gpu::GpuIndexIVFFlat search_index_2(&faiss_resources_2, num_dims, kNumCells,faiss::METRIC_L2, faiss_config_2);

        // faiss::gpu::GpuIndexIVFPQ search_index(&faiss_resources, num_dims, kNumCells, kSubQuant, kBPC, faiss::METRIC_L2, faiss_config);
        // faiss::gpu::GpuIndexIVFPQ search_index_2(&faiss_resources, num_dims, kNumCells, kSubQuant, kBPC, faiss::METRIC_L2, faiss_config_2);
        search_index.setNumProbes(kNumCellsToProbe);
        // search_index_2.setNumProbes(kNumCellsToProbe);

        faiss::gpu::IndexProxy search_proxy;
        search_proxy.addIndex(&search_index);
        // search_proxy.addIndex(&search_index_2);

        // Add the points to the index
        // search_index.train(num_points, points);
        // search_index_2.train(num_points, points);
        // auto h1 = std::async(std::launch::async, &faiss::gpu::GpuIndexIVFFlat::train, &search_index, num_points, points);
        // auto h2 = std::async(std::launch::async, &faiss::gpu::GpuIndexIVFFlat::train, &search_index_2, num_points, points);
        search_proxy.train(num_points, points);
        // h1.get();
        // h2.get();
        // search_index.add(num_points, points);
        // search_index_2.add(num_points, points);
        // search_proxy.add(num_points, points);
        // h1 = std::async(std::launch::async, &faiss::gpu::GpuIndexIVFFlat::add, &search_index, num_points, points);
        // h2 = std::async(std::launch::async, &faiss::gpu::GpuIndexIVFFlat::add, &search_index_2, num_points, points);
        search_proxy.add(num_points, points);
        // h1.get();
        // h2.get();

        // Perform the KNN query
        // auto h1 = std::async(std::launch::async, &faiss::gpu::GpuIndexIVFFlat::search, &search_index, num_points/2, points, num_near_neighbors, distances, indices);
        // auto h2 = std::async(std::launch::async, &faiss::gpu::GpuIndexIVFFlat::search, &search_index_2, num_points - num_points/2, points + num_dims*(num_points/2), num_near_neighbors, distances + num_near_neighbors*(num_points/2), indices + num_near_neighbors*(num_points/2));
        // search_index_2.search(num_points - num_points/2, points + num_dims*(num_points/2), num_near_neighbors, distances + num_near_neighbors*(num_points/2), indices*(num_points/2));

        // h1.get();
        // h2.get();
        search_proxy.search(num_points, points, num_near_neighbors, distances, indices);
    }
    else if (num_near_neighbors < 1024) {
        std::cout << "Starting NN calculation..." << std::endl;
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

// // Construct the GPU resources necessary
// faiss::gpu::StandardGpuResources faiss_resources;
// faiss_resources.noTempMemory();

// // Construct the GPU configuration object
// faiss::gpu::GpuIndexIVFPQConfig faiss_config;

// // // TODO(David): Allow for dynamic device placement
// faiss_config.device = 0;

// faiss_config.indicesOptions = faiss::gpu::INDICES_32_BIT;
// faiss_config.flatConfig.useFloat16 = false;
// faiss_config.usePrecomputedTables = true;
// // faiss_config.useFloat16IVFStorage = false;

// faiss::gpu::GpuIndexIVFPQ search_index(&faiss_resources, num_dims, kNumCells, kSubQuant, kBPC, faiss::METRIC_L2, faiss_config);
// search_index.setNumProbes(kNumCellsToProbe);

// // Add the points to the index
// search_index.train(num_points, points);
// search_index.add(num_points, points);

// // Perform the KNN query
// search_index.search(num_points, points, num_near_neighbors,
//                         distances, indices);


// // faiss::gpu::GpuIndexIVFFlat search_index(&faiss_resources,
// //         num_dims, kNumCells, faiss::METRIC_L2, faiss_config);
// // search_index.setNumProbes(kNumCellsToProbe);
// // search_index.train(num_points, points);
// // search_index.add(num_points, points);

// // // Perform the KNN query
// // search_index.search(num_points, points, num_near_neighbors,
// //                     distances, indices);

// TODO: Add -1 notification here... and how to deal with it if it happens
// TODO: Maybe think about getting FAISS to return integers (long-term todo)
__global__
void tsnecuda::util::PostprocessNeighborIndicesKernel(
                                    volatile int * __restrict__ indices,
                                    const long * __restrict__ long_indices,
                                    const int num_points,
                                    const int num_neighbors)
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_points * num_neighbors) return;
    // Set pij to 0 for each of the broken values - Note: this should be handled in the ComputePijKernel now
    // if (matrix[TID] == 1.0f) matrix[TID] = 0.0f;
    indices[TID] = (int) long_indices[TID];
}

void tsnecuda::util::PostprocessNeighborIndices(
                tsnecuda::GpuOptions &gpu_opt,
                thrust::device_vector<int> &indices,
                thrust::device_vector<int64_t> &long_indices,
                const int num_points,
                const int num_neighbors
        )
{
    const int num_threads = 128;
    const int num_blocks = iDivUp(num_points*num_neighbors, num_threads);
    tsnecuda::util::PostprocessNeighborIndicesKernel<<<num_blocks, num_threads>>>(
                        thrust::raw_pointer_cast(indices.data()),
                        thrust::raw_pointer_cast(long_indices.data()),
                        num_points,
                        num_neighbors
            );
    GpuErrorCheck(cudaDeviceSynchronize());
}
