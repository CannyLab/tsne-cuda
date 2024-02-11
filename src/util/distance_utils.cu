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
    const float *__restrict__ d_squared_norms,
    float *__restrict__ d_dot_products,
    const int num_points)
{
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
                                             const int num_dims)
{
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
                                      const int num_dims)
{
    tsnecuda::util::SquaredPairwiseDistance(handle, d_distances, d_points,
                                            num_points, num_dims);
    tsnecuda::util::SqrtDeviceVector(d_distances, d_distances);
}

void tsnecuda::util::KNearestNeighbors(tsnecuda::GpuOptions &gpu_opt,
                                       tsnecuda::Options &base_options,
                                       int64_t *indices, float *distances,
                                       const float *const points, const int num_dims,
                                       const int num_points, const int num_near_neighbors)
{
    const int32_t kNumCells = static_cast<int32_t>(
        std::sqrt(static_cast<float>(num_points)));
    const int32_t kNumCellsToProbe = 20;

    // Construct the CPU version of the index
    faiss::IndexFlatL2 quantizer(num_dims);
    faiss::IndexIVFFlat cpu_index(&quantizer, num_dims, kNumCells, faiss::METRIC_L2);
    cpu_index.nprobe = kNumCellsToProbe;

    if (num_near_neighbors < 1024)
    {
        int ngpus = faiss::gpu::getNumDevices();
        std::vector<faiss::gpu::GpuResourcesProvider *> res;
        std::vector<int> devs;
        for (int i = 0; i < ngpus; i++)
        {
            res.push_back(new faiss::gpu::StandardGpuResources);
            devs.push_back(i);
        }

        // Convert the CPU index to GPU index
        faiss::Index *search_index = faiss::gpu::index_cpu_to_gpu_multiple(res, devs, &cpu_index);

        search_index->train(num_points, points);
        search_index->add(num_points, points);
        search_index->search(num_points, points, num_near_neighbors, distances, indices);

        delete search_index;
        for (int i = 0; i < ngpus; i++)
        {
            delete res[i];
        }
    }
    else
    {
        // Construct the index table on the CPU (since the GPU
        // can only handle 1023 neighbors)
        cpu_index.train(num_points, points);
        cpu_index.add(num_points, points);
        // Perform the KNN query
        cpu_index.search(num_points, points, num_near_neighbors,
                         distances, indices);
    }
}

// TODO: Add -1 notification here... and how to deal with it if it happens
// TODO: Maybe think about getting FAISS to return integers (long-term todo)
__global__ void tsnecuda::util::PostprocessNeighborIndicesKernel(
    volatile int *__restrict__ indices,
    const int64_t *__restrict__ long_indices,
    const int num_points,
    const int num_neighbors)
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_points * num_neighbors)
        return;
    // Set pij to 0 for each of the broken values - Note: this should be handled in the ComputePijKernel now
    // if (matrix[TID] == 1.0f) matrix[TID] = 0.0f;
    indices[TID] = (int)long_indices[TID];
}

void tsnecuda::util::PostprocessNeighborIndices(
    tsnecuda::GpuOptions &gpu_opt,
    thrust::device_vector<int> &indices,
    thrust::device_vector<int64_t> &long_indices,
    const int num_points,
    const int num_neighbors)
{
    const int num_threads = 128;
    const int num_blocks = iDivUp(num_points * num_neighbors, num_threads);
    tsnecuda::util::PostprocessNeighborIndicesKernel<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(long_indices.data()),
        num_points,
        num_neighbors);
    GpuErrorCheck(cudaDeviceSynchronize());
}
