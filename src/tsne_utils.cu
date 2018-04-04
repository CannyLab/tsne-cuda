#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <random>
#include <sys/time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <stdexcept>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#include "tsne_utils.cuh"
#include "Utilities.cuh"

#define PROJDIM 2

struct func_square {
	__host__ __device__ double operator()(const float &x) const { return x * x; }
};

struct func_sqrt {
    __host__ __device__ double operator()(const float &x) const { return pow(x, 0.5); }
};

struct func_exp {
    __host__ __device__ double operator()(const float &x) const { return exp(x); }
};

struct func_exp_no_zero {
    __host__ __device__ double operator()(const float &x) const { return x < -1e-4 ? exp(x) : 0; }
};

struct func_inv {
    __host__ __device__ double operator()(const float &x) const { return pow(x, -1.0); }
};

struct func_inc {
    __host__ __device__ double operator()(const float &x) const { return x + 1; }
};

struct func_inc_inv {
    __host__ __device__ double operator()(const float &x) const { return pow(x + 1, -1.0); }
};

struct func_inc_inv_ignore_zero {
    __host__ __device__ double operator()(const float &x) const { return x > 1e-4 ? pow(x + 1, -1.0) : 0; }
};

struct func_ln {
    __host__ __device__ double operator()(const float &x) const { return log(x); }
};

struct func_kl {
    __host__ __device__ double operator()(const float &x, const float &y) const { return y < 1e-4 ? 0 : x * log(x / y); }
};

struct func_abs {
    __host__ __device__ double operator()(const float &x) const { return fabs(x); }
};

struct prg
{
    float a, b;

    __host__ __device__
    prg(float _a=-1.f, float _b=1.f) : a(_a), b(_b) {};

    __host__ __device__
        float operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(a, b);
            rng.discard(n);

            return dist(rng);
        }
};

thrust::device_vector<float> random_vector(const unsigned int N) {
    thrust::device_vector<float> vector(N);
    thrust::counting_iterator<float> first(0.0f);
    thrust::counting_iterator<float> last = first + N;
    thrust::transform(first, last, vector.begin(), prg(-10.0f, 10.0f));
    return vector;
}

// Performs the operation matrix[i, :] = binary_op(matrix[i, :], alpha * vector) for each row i in the matrix
template<typename BinaryFunction, typename T>
__global__ void _broadcast_row_vec(
                                    T * __restrict__ matrix, 
                                    const T * __restrict__ vector, 
                                    const unsigned int N, 
                                    const unsigned int M, 
                                    BinaryFunction binary_op, 
                                    const T alpha) 
{
    const unsigned int TID = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int i = TID % N;
    const unsigned int j = TID / N;

    if (j < M) matrix[j * N + i] = binary_op(matrix[j * N + i], alpha * vector[j]);
}

// Performs the operation matrix[:, j] = binary_op(matrix[:, j], alpha * vector) for each col i in the matrix
template<typename BinaryFunction, typename T>
__global__ void _broadcast_col_vec(
                                    T * __restrict__ matrix, 
                                    const T * __restrict__ vector, 
                                    const unsigned int N, 
                                    const unsigned int M,
                                    BinaryFunction binary_op,
                                    const T alpha)
{
     const unsigned int TID = threadIdx.x + blockIdx.x * blockDim.x;
     const unsigned int i = TID % N;
     const unsigned int j = TID / N;

     if (j < M) matrix[j * N + i] = binary_op(matrix[j * N + i], alpha * vector[i]);
}

/**
 * Inputs: 
 *      matrix => (N x M) matrix stored in column major order
 *      vector => Length N vector if axis == 0, length M vector if axis == 1
 *      N, M => dimensions of matrix
 *      binary_op => an operation that takes in two arguments of type T and returns a type T
 *      axis => 0 or 1, controlls whether this runs a column or row broadcast
 *      alpha => scalar multiple for vector
 *
 * Notes:
 *      should axis == 0 be row or column broadcasting? and vice versa for axis == 1?
 */
template<typename BinaryFunction, typename T>
void broadcast_matrix_vector(
                            thrust::device_vector<T> &matrix, 
                            const thrust::device_vector<T> &vector, 
                            const unsigned int N, 
                            const unsigned int M, 
                            BinaryFunction binary_op,
                            const unsigned int axis,
                            const T alpha) 
{ 
    // Checks to make sure dimensions are correct
    assert(matrix.size() >= N * M);
    assert((axis == 0 && vector.size() >= N) || (axis == 1 && vector.size() >= M));
    
    const unsigned int BLOCKSIZE = 32;
    const unsigned int NBLOCKS = iDivUp(N * M, BLOCKSIZE);
    if (axis == 0) {
        _broadcast_col_vec<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(matrix.data()),
                                                    thrust::raw_pointer_cast(vector.data()), 
                                                    N, M, binary_op, alpha);
    } else {
        _broadcast_row_vec<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(matrix.data()),
                                                    thrust::raw_pointer_cast(vector.data()), 
                                                    N, M, binary_op, alpha);
    }
}

__global__ void assemble_final_result(const float * __restrict__ d_norms_x_2, float * __restrict__ d_dots,
									  const int N) {

	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < N) && (j < N)) d_dots[i * N + j] = d_norms_x_2[j] + d_norms_x_2[i] - 2 * d_dots[i * N + j];
    
}

void printarray(thrust::device_vector<float> vec, const unsigned int N, const unsigned int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << vec[i + j * N] << " ";
        }
        std::cout << std::endl;
    }
}

// Code from https://github.com/OrangeOwlSolutions/cuBLAS/blob/master/All_pairs_distances.cu
// Expects N x NDIMS matrix in points
void pairwise_dist(cublasHandle_t &handle, thrust::device_vector<float> &distances, const thrust::device_vector<float> &points, const unsigned int N, const unsigned int NDIMS) {
    const unsigned int BLOCKSIZE = 16;

    auto squared_vals = square(points, N * NDIMS);
    auto squared_norms = reduce_sum(handle, squared_vals, N, NDIMS, 1);
    
    float alpha = 1.f;
    float beta = 0.f;
    // Could replace this with cublasSsyrk, might be faster?
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, NDIMS, &alpha,
		                       thrust::raw_pointer_cast(points.data()), N, thrust::raw_pointer_cast(points.data()), N, &beta,
							   thrust::raw_pointer_cast(distances.data()), N));
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(iDivUp(N, BLOCKSIZE), iDivUp(N, BLOCKSIZE));
	assemble_final_result<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(squared_norms.data()), 
		                                         thrust::raw_pointer_cast(distances.data()), N);
}

void gauss_normalize(cublasHandle_t &handle, thrust::device_vector<float> &points, const unsigned int N, const unsigned int NDIMS) {
    auto means = reduce_mean(handle, points, N, NDIMS, 0);

    // zero center
    broadcast_matrix_vector(points, means, N, NDIMS, thrust::minus<float>(), 1, 1.f);
    
    // compute standard deviation
    auto squared_vals = square(points, N * NDIMS);
    auto norm_sum_of_squares = reduce_alpha(handle, squared_vals, N, NDIMS, 1.f / (N - 1), 0);
    auto stddev = sqrt(norm_sum_of_squares, N * NDIMS);

    // normalize
    broadcast_matrix_vector(points, stddev, N, NDIMS, thrust::divides<float>(), 1, 1.f);
}

thrust::device_vector<float> square(const thrust::device_vector<float> &vec, const unsigned int N) {
    thrust::device_vector<float> squared_vals(N);
    thrust::transform(vec.begin(), vec.end(), squared_vals.begin(), func_square());
    return squared_vals;
}

thrust::device_vector<float> sqrt(const thrust::device_vector<float> &vec, const unsigned int N) {
    thrust::device_vector<float> sqrt_vals(N);
    thrust::transform(vec.begin(), vec.end(), sqrt_vals.begin(), func_sqrt());
    return sqrt_vals;
}

void max_norm(thrust::device_vector<float> &vec) {
    float max_val = thrust::transform_reduce(vec.begin(), vec.end(), func_abs(), 0.0f, thrust::maximum<float>());
    thrust::constant_iterator<float> div_iter(max_val);
    
    thrust::transform(vec.begin(), vec.end(), div_iter, vec.begin(), thrust::divides<float>());
}

thrust::device_vector<float> compute_pij(cublasHandle_t &handle, thrust::device_vector<float> &points, thrust::device_vector<float> &sigma, const unsigned int N, const unsigned int NDIMS) {
    thrust::device_vector<float> pij_vals(N * N);
    pairwise_dist(handle, pij_vals, points, N, NDIMS);
    auto sigma_squared = square(sigma, N);
    
    broadcast_matrix_vector(pij_vals, sigma_squared, N, N, thrust::divides<float>(), 1, -2.0f);

    // exponentiate - func_exp_no_zeros() ignores values > -1e-4
    thrust::transform(pij_vals.begin(), pij_vals.end(), pij_vals.begin(), func_exp_no_zero());
    // reduce_sum over rows
    auto sums = reduce_sum(handle, pij_vals, N, N, 1);
    // divide column by resulting vector
    broadcast_matrix_vector(pij_vals, sums, N, N, thrust::divides<float>(), 0, 1.0f);

    float alpha = 0.5f/N;
    float beta = 0.5f/N;
    thrust::device_vector<float> pij_output(N*N);
    cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, &alpha, thrust::raw_pointer_cast(pij_vals.data()), N, 
                               &beta, thrust::raw_pointer_cast(pij_vals.data()), N, thrust::raw_pointer_cast(pij_output.data()), N));

    return pij_output;
}

float compute_gradients(cublasHandle_t &handle, 
                        thrust::device_vector<float> &forces,
                        thrust::device_vector<float> &dist, 
                        thrust::device_vector<float> &ys, 
                        thrust::device_vector<float> &pij, 
                        thrust::device_vector<float> &qij, 
                        const unsigned int N,
                        float eta) 
{
    pairwise_dist(handle, dist, ys, N, PROJDIM);
    // dist = (1 + ||y_i - y_j||^2)^-1
    thrust::transform(dist.begin(), dist.end(), dist.begin(), func_inc_inv_ignore_zero());
    // printarray(dist, N, N);
    auto sums = reduce_sum(handle, dist, N, N, 1);
    // printarray(sums, 1, N);
    thrust::copy(dist.begin(), dist.end(), qij.begin());

    // qij = (1 + ||y_i - y_j||^2)^-1 / \Sum_{k != i} (1 + ||y_i - y_k||^2)^-1
    broadcast_matrix_vector(qij, sums, N, N, thrust::divides<float>(), 0, 1.0f);
    // printarray(qij, N, N);
    thrust::device_vector<float> loss_(N * N);
    thrust::transform(pij.begin(), pij.end(), qij.begin(), loss_.begin(), func_kl());

    float loss = thrust::reduce(loss_.begin(), loss_.end(), 5.0f, thrust::minimum<float>());

    thrust::transform(pij.begin(), pij.end(), qij.begin(), qij.begin(), thrust::minus<float>());
    thrust::transform(qij.begin(), qij.end(), dist.begin(), qij.begin(), thrust::multiplies<float>());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    thrust::device_vector<float> ones(N, 1.0f);
    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, thrust::raw_pointer_cast(qij.data()), N,
                                thrust::raw_pointer_cast(ones.data()), 1, &beta, thrust::raw_pointer_cast(forces.data()), 1));

    // TODO: needs to change for 3 dimensions
    thrust::copy(forces.begin(), forces.begin() + N, forces.begin() + N);

    // forces = A * ones(N, 1) .* ys
    thrust::transform(forces.begin(), forces.end(), ys.begin(), forces.begin(), thrust::multiplies<float>());

    alpha = -4.0f * eta;
    beta = 4.0f * eta;
    // TODO: needs to change for 3 dimensions
    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, thrust::raw_pointer_cast(qij.data()), N,
                                thrust::raw_pointer_cast(ys.data()), 1, &beta, thrust::raw_pointer_cast(forces.data()), 1));
    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, thrust::raw_pointer_cast(qij.data()), N,
                                thrust::raw_pointer_cast(ys.data() + N), 1, &beta, thrust::raw_pointer_cast(forces.data() + N), 1));

    return loss;
}

// expects matrix of size N x M
thrust::device_vector<float> reduce_alpha(cublasHandle_t &handle, const thrust::device_vector<float> &matrix, const unsigned int N, const unsigned int M, float alpha, const int axis) {
    if (axis == 0) {
        thrust::device_vector<float> ones(N, 1.f);
        thrust::device_vector<float> means(M);

        float beta = 0.f;
        cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, thrust::raw_pointer_cast(matrix.data()), N,
                                    thrust::raw_pointer_cast(ones.data()), 1, &beta, thrust::raw_pointer_cast(means.data()), 1));
        return means;
    } else if (axis == 1) {
        thrust::device_vector<float> ones(M, 1.f);
        thrust::device_vector<float> means(N);

        float beta = 0.f;
        cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, thrust::raw_pointer_cast(matrix.data()), N,
                                    thrust::raw_pointer_cast(ones.data()), 1, &beta, thrust::raw_pointer_cast(means.data()), 1));
        return means;
    } else {
        throw std::runtime_error("Axis must be 0 or 1.");
    }
}

thrust::device_vector<float> reduce_mean(cublasHandle_t &handle, const thrust::device_vector<float> &matrix, const unsigned int N, const unsigned int M, const int axis) {
    float alpha = 1.f / N;
    return reduce_alpha(handle, matrix, N, M, alpha, axis);
}


thrust::device_vector<float> reduce_sum(cublasHandle_t &handle, const thrust::device_vector<float> &matrix, const unsigned int N, const unsigned int M, const int axis) {
    float alpha = 1.f;
    return reduce_alpha(handle, matrix, N, M, alpha, axis);
}

thrust::device_vector<float> do_tsne(cublasHandle_t &handle, thrust::device_vector<float> &points, const unsigned int N, const unsigned int NDIMS) {
    max_norm(points);
    thrust::device_vector<float> sigmas(N, 0.5f);
    auto pij = compute_pij(handle, points, sigmas, N, NDIMS);
    thrust::device_vector<float> forces(N * PROJDIM);
    thrust::device_vector<float> ys = random_vector(N * PROJDIM);
    thrust::device_vector<float> qij(N * N);
    thrust::device_vector<float> dist(N * N);
    float eta = 1e-2f;
    float loss, prevloss = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 1000; i++) {
        loss = compute_gradients(handle, forces, dist, ys, pij, qij, N, eta);
        thrust::transform(ys.begin(), ys.end(), forces.begin(), ys.begin(), thrust::plus<float>());
        if (loss > prevloss)
            eta /= 2.;
        if (i % 10 == 0)
            std::cout << "Iteration: " << i << ", Loss: " << loss << ", ForceMag: " << thrust::reduce(forces.begin(), forces.end(), 0.0f, thrust::plus<float>()) << std::endl;
        prevloss = loss;
    }
    return ys;
}

template <typename BinaryFunction>
void test_func(BinaryFunction binary_op) {
    thrust::device_vector<float> matrix(4 * 5, 5.0f);
    thrust::device_vector<float> twos(5, 2.0f);
    broadcast_matrix_vector(matrix, twos, 4, 5, binary_op, 0, 1.0f);
    printarray(matrix, 4, 5);
}


int main(int argc, char **argv) {
    const unsigned int NDIMS = 50;
    const unsigned int N = 1 << 14;
    
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrices allocation and initialization
    thrust::device_vector<float> d_X(NDIMS * N);
    for (size_t i = 0; i < d_X.size(); i++) 
        d_X[i] = (float) dist(rng);

    thrust::device_vector<float> sigmas(N, 1.0f);
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Starting TSNE calculation with %u points.\n", N);
    cudaEventRecord(start);
    do_tsne(handle, d_X, N, NDIMS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f (ms)\n", milliseconds);
    return 0;
}

int run_tests(int argc, char **argv) {
    test_func(thrust::plus<float>());
    test_func(thrust::multiplies<float>());
    test_func(thrust::divides<float>());
    return 0;
}

