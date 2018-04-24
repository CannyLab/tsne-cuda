/**
 * @brief Implementation of different distances
 * 
 * @file distance_utils.cu
 * @author David Chan
 * @date 2018-04-04
 */

 #include "util/distance_utils.h"

struct func_sqrt {
    __host__ __device__ float operator()(const float &x) const { return pow(x, 0.5); }
};

// This really does a simultaneous row/col matrix vector broadcast to compute ||x^2|| + ||y^2|| - 2 x^Ty.
// Added fabs to deal with numerical instabilities. I think this is a reasonable solution
 __global__ void Distance::assemble_final_result(const float * __restrict__ d_norms_x_2, 
                                       float * __restrict__ d_dots,
                                       const int N)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        const int j = threadIdx.y + blockIdx.y * blockDim.y;

        if ((i < N) && (j < N))
            d_dots[i * N + j] = fabs(d_norms_x_2[j] + d_norms_x_2[i] - 2 * d_dots[i * N + j]);
    }
// Code from https://github.com/OrangeOwlSolutions/cuBLAS/blob/master/All_pairs_distances.cu
// Expects N x NDIMS matrix in points
// Squared norms taken from diagnoal of dot product which should be faster and result in actually zeroing out the diagonal in assemble_final_result
void Distance::squared_pairwise_dist(cublasHandle_t &handle, 
                   thrust::device_vector<float> &distances, 
                   const thrust::device_vector<float> &points, 
                   const unsigned int N, 
                   const unsigned int NDIMS) 
{
    const unsigned int BLOCKSIZE = 16;
    // thrust::device_vector<float> squared_vals(points.size());
    // square(points, squared_vals);
    // auto squared_norms = Reduce::reduce_sum(handle, squared_vals, N, NDIMS, 1);
    
    float alpha = 1.f;
    float beta = 0.f;
    // Could replace this with cublasSsyrk, might be faster?
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, NDIMS, &alpha,
		                       thrust::raw_pointer_cast(points.data()), N, thrust::raw_pointer_cast(points.data()), N, &beta,
							   thrust::raw_pointer_cast(distances.data()), N));
  
    typedef thrust::device_vector<float>::iterator Iterator;
    strided_range<Iterator> diag(distances.begin(), distances.end(), N + 1);
    thrust::device_vector<float> squared_norms(N);
    thrust::copy(diag.begin(), diag.end(), squared_norms.begin());

	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(iDivUp(N, BLOCKSIZE), iDivUp(N, BLOCKSIZE));
	Distance::assemble_final_result<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(squared_norms.data()), 
                                                 thrust::raw_pointer_cast(distances.data()), N);
                                                 
}

void Distance::pairwise_dist(cublasHandle_t &handle, 
                   thrust::device_vector<float> &distances, 
                   const thrust::device_vector<float> &points, 
                   const unsigned int N, 
                   const unsigned int NDIMS) 
{
    Distance::squared_pairwise_dist(handle, distances, points, N, NDIMS);
    Math::sqrt(distances, distances);
}

void Distance::knn(float* points, long* I, float* D, const unsigned int N_DIM, const unsigned int N_POINTS, const unsigned int K) {
    const int nlist = (int) std::sqrt((float)N_POINTS);
    const int nprobe = 5;
    
    if (K < 1024) {
        // Construct the GPU resources necessary
        faiss::gpu::StandardGpuResources res;
        res.noTempMemory();

        // Construct the GPU configuration object
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        config.indicesOptions = faiss::gpu::INDICES_32_BIT;
        config.flatConfig.useFloat16 = true;
        config.useFloat16IVFStorage = true;

        faiss::gpu::GpuIndexIVFFlat index(&res, N_DIM, nlist, faiss::METRIC_L2, config);
        index.setNumProbes(nprobe);
        index.train(N_POINTS, points);
        index.add(N_POINTS, points);

        // Perform the KNN query
        index.search(N_POINTS, points, K, D, I);
    } else {
        // Construct the index table on the CPU (since the GPU can only handle 1023 neighbors)
        faiss::IndexFlatL2 quantizer(N_DIM);
        faiss::IndexIVFFlat index(&quantizer, N_DIM, nlist, faiss::METRIC_L2); // We can probably change the metric later
        index.train(N_POINTS, points);
        index.add(N_POINTS, points);

        // Perform the KNN query
        index.nprobe = nprobe;
        index.search(N_POINTS, points, K, D, I);
    }
}
