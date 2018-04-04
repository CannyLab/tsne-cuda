/**
 * @brief Implementation of naive T-SNE
 * 
 * @file naive_tsne.cu
 * @author David Chan
 * @date 2018-04-04
 */

 #include "naive_tsne.h"

thrust::device_vector<float> compute_pij(cublasHandle_t &handle, 
                                         thrust::device_vector<float> &points, 
                                         thrust::device_vector<float> &sigma, 
                                         const unsigned int N, 
                                         const unsigned int NDIMS) 
{
    thrust::device_vector<float> pij_vals(N * N);
    pairwise_dist(handle, pij_vals, points, N, NDIMS);
    auto sigma_squared = square(sigma, N);
    
    broadcast_matrix_vector(pij_vals, sigma_squared, N, N, thrust::divides<float>(), 1, -2.0f);

    // exponentiate - func_exp_no_zeros() ignores values > -1e-4
    thrust::transform(pij_vals.begin(), pij_vals.end(), pij_vals.begin(), tfunc::exp_no_zero());
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
    thrust::transform(dist.begin(), dist.end(), dist.begin(), tfunc::inc_inv_ignore_zero());
    // printarray(dist, N, N);
    auto sums = reduce_sum(handle, dist, N, N, 1);
    // printarray(sums, 1, N);
    thrust::copy(dist.begin(), dist.end(), qij.begin());

    // qij = (1 + ||y_i - y_j||^2)^-1 / \Sum_{k != i} (1 + ||y_i - y_k||^2)^-1
    broadcast_matrix_vector(qij, sums, N, N, thrust::divides<float>(), 0, 1.0f);
    // printarray(qij, N, N);
    thrust::device_vector<float> loss_(N * N);
    thrust::transform(pij.begin(), pij.end(), qij.begin(), loss_.begin(), tfunc::kl());

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

thrust::device_vector<float> naive_tsne(cublasHandle_t &handle, 
                                        thrust::device_vector<float> &points, 
                                        const unsigned int N, 
                                        const unsigned int NDIMS)
{
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

