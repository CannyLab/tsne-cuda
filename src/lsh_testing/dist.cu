
// Basic includes
#include <iostream>
#include <random>
#include <cuda.h>
#include <iomanip>
#include <stdlib.h>    
#include <time.h>      

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void compute_hashes(float* points, float* projections, ulong2* us_hashes, const int M, const int N_DIM, const int N, const int OFF) {
    int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= N)
        return;
        
    // Compute the projection for each point
    unsigned long hash_val = 0;
    for (int i = 0; i < M; i++) {
        float v = 0;
        for (int d = 0; d < N_DIM; d++) {
            v += projections[d + i*M] * points[d + TID*N_DIM];
        }
        if (v > 0) {
            hash_val |= 1;
        }
        hash_val <<= 1;
    }
    us_hashes[TID] = make_ulong2(hash_val, OFF + TID);
}

__global__ void pairwise_similarity(ulong2* hashes_x, ulong2* hashes_y, float* major_block, const int N_X, const int N_Y) {
    int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= N_X*N_Y) {
        return;
    }
        
    // Compute your xor/distance
    unsigned long val = (hashes_x[TID / N_Y].x) ^ (hashes_y[TID % N_X].x);
    // Count the number of 1s in the hash
    float count = 0.0;
    while (val > 0) {
        count += (val & 0x1) ? 1.0 : 0.0;
        val >>= 1;
    }
    major_block[TID] = count/64.0;
}



int main(int argc, char** argv) {

    // Generate random uniform points
    std::default_random_engine generator(time(NULL));
    std::uniform_real_distribution<float> distribution(0.0,1.0);

    // Allocate memory for our points array
    const int N_POINTS = 16;
    const int N_DIM = 1000;
    float* input_point_data = new float[N_POINTS * N_DIM];

    // Get some random points
    for (int i = 0; i < N_POINTS; i += 1) {
        std::normal_distribution<float> normal(distribution(generator),1.0);
        for (int k = 0; k < N_DIM; k++) {
            input_point_data[i*N_DIM + k] = normal(generator);
        }
    }

    // Ok, now we do the LSH stuff. First, compute all of the hashes

    // How many particles we want to handle at once
    const int POINT_BLOCK_SIZE = 8;
    const int N_PROJS = 64;
    
    // Allocate the on-device point buffers (we're going to double-buffer)
    // so it's all good
    float* d_points_buffer;
    cudaMalloc((void **) &d_points_buffer, POINT_BLOCK_SIZE * sizeof(float) * N_DIM);
    float* d_projections;
    cudaMalloc((void **) &d_projections, N_PROJS * sizeof(float) * N_DIM);
    ulong2* d_hashes;
    cudaMalloc((void **) &d_hashes, POINT_BLOCK_SIZE * sizeof(ulong2));

    // Construct the output vector
    ulong2* host_hashes = new ulong2[N_POINTS];

    // Construct the projections and copy them to the GPU
    std::normal_distribution<float> normal(0.0,1.0);
    float* host_projections = new float[N_DIM*N_PROJS];
    for (int i = 0; i < N_PROJS*N_DIM; i++)
        host_projections[i] = normal(generator);

    cudaMemcpy(d_projections, host_projections, N_PROJS*N_DIM*sizeof(float), cudaMemcpyHostToDevice);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Compute the hashes
    for (int i = 0; i < N_POINTS; i+=POINT_BLOCK_SIZE) {
        // Copy the current block of points to the gpu
        const int N = std::min<int>(POINT_BLOCK_SIZE, N_POINTS-i);
        cudaMemcpy(d_points_buffer, &(input_point_data[i*N_DIM]), N_DIM*N*sizeof(float),cudaMemcpyHostToDevice);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // Compute the hashes on the gpu
        const int THREADS = 256;
        const int BLOCKS = (N + THREADS-1) / THREADS;
        compute_hashes<<<BLOCKS, THREADS>>>(d_points_buffer, d_projections, d_hashes, N_PROJS, N_DIM, N, i);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // Copy the hashes back
        cudaMemcpy(&(host_hashes[i]), d_hashes, N*sizeof(ulong2), cudaMemcpyDeviceToHost);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMemset(d_points_buffer, 0, POINT_BLOCK_SIZE*N_DIM*sizeof(float));
    }

    cudaFree(d_points_buffer);
    cudaFree(d_projections);
    cudaFree(d_hashes);

    for (int i = 0; i < N_POINTS; i++) {
        std::cout << host_hashes[i].x <<"," << host_hashes[i].y << std::endl;
    }

    // Allocate memory on the GPU for the hashes
    const int BLOCK_X = 4;
    const int BLOCK_Y = 4;
    ulong2* d_xhash_buffer;
    cudaMalloc((void **) &d_xhash_buffer, BLOCK_X * sizeof(ulong2));
    ulong2* d_yhash_buffer;
    cudaMalloc((void **) &d_yhash_buffer, BLOCK_Y * sizeof(ulong2));
    float* d_similarity_buffer;
    cudaMalloc((void **) &d_similarity_buffer, BLOCK_X * BLOCK_Y * sizeof(float));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Allocate CPU similarity output
    float* point_rel_distances = new float[N_POINTS*N_POINTS];

    // Compute the pairwise xor of the values
    for (int x = 0; x < N_POINTS; x+= BLOCK_X) {

        const int X = std::min<int>(BLOCK_X, N_POINTS - x);
        std::cout << "x: " << x << " " << X << std::endl;
        
        // Copy hashes to the GPU
        cudaMemcpy(d_xhash_buffer, &(host_hashes[x]), X*sizeof(ulong2), cudaMemcpyHostToDevice);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        
        for (int y = 0; y < N_POINTS; y+=BLOCK_Y) {
            const int Y = std::min<int>(BLOCK_Y, N_POINTS - y);

            std::cout << "y: " << y << " " << Y << std::endl;

            // Copy the hashes to the GPU
            cudaMemcpy(d_yhash_buffer, &(host_hashes[y]), Y*sizeof(ulong2), cudaMemcpyHostToDevice);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            // Compute the similarity
            const int THREADS = 256;
            const int BLOCKS = (X*Y + THREADS-1) / THREADS;
            pairwise_similarity<<<BLOCKS, THREADS>>>(d_xhash_buffer, d_yhash_buffer, d_similarity_buffer, X, Y);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            // Copy the point matrix back
            for (int a = 0; a < X; a++) {
                cudaMemcpy(&(point_rel_distances[(a+x)*N_POINTS + y]), &(d_similarity_buffer[a*Y]), Y*sizeof(float), cudaMemcpyDeviceToHost);
            }
                
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }
    }

    for (int j = 0; j < N_POINTS; j++) {
        for (int i = 0; i < N_POINTS; i++) {
            std::cout << std::fixed << std::setprecision(3) << 1 - point_rel_distances[i + j*N_POINTS]/3.1415926525 << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << std::endl; std::cout << std::endl << std::endl;;
    float* diff = new float[N_POINTS * N_POINTS];
    for (int j = 0; j < N_POINTS; j++) {
        for (int i = 0; i < N_POINTS; i++) {

            // Compute the true distance between i and j
            float val = 0.0;
            float mag_a = 0.0;
            float mag_b = 0.0;
            for (int k = 0; k < N_DIM; k++) {
                val += (input_point_data[i*N_DIM + k] * input_point_data[j*N_DIM + k]);
                mag_a += input_point_data[i*N_DIM + k] * input_point_data[i*N_DIM + k];
                mag_b += input_point_data[j*N_DIM + k] * input_point_data[j*N_DIM + k];
            }
            diff[i + j*N_POINTS] = abs((1 - point_rel_distances[i + j*N_POINTS]/3.1415926525) - (val/(sqrt(mag_a)*sqrt(mag_b))));
            std::cout << std::fixed << std::setprecision(3) << val/(sqrt(mag_a)*sqrt(mag_b)) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << std::endl;
    for (int j = 0; j < N_POINTS; j++) {
        for (int i = 0; i < N_POINTS; i++) {
            std::cout << std::fixed << std::setprecision(3) << diff[i + j*N_POINTS] << " ";
        }
        std::cout << std::endl;
    }
    


    

    std::cout << "Done!" << std::endl;
    
    return 0;
}