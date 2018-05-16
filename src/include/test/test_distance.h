/**
 * @brief Unit Tests for the distance functions
 * 
 * @file test_distance.h
 * @author David Chan
 * @date 2018-04-11
 */

void test_pairwise_distance(int N, int NDIM) {
    //srand (time(NULL));

    // Create some random points in 2 dimensions
    float points[N * NDIM];
    for (int i = 0; i < N*NDIM; i++) {
        points[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    // Construct the thrust vector
    thrust::host_vector<float> h_points(N*NDIM);
    for (int i = 0; i < N*NDIM; i++) 
        h_points[i] = points[i]; 
    thrust::device_vector<float> d_points(N*NDIM);
    thrust::copy(h_points.begin(), h_points.end(), d_points.begin());
    thrust::device_vector<float> d_distances(N*N);

    // Construct the CUBLAS handle
    cublasHandle_t handle;
    CublasSafeCall(cublasCreate(&handle));
    tsnecuda::util::PairwiseDistance(handle, d_distances, d_points, N, NDIM);

    thrust::host_vector<float> h_distances(N*N);
    thrust::copy(d_distances.begin(), d_distances.end(), h_distances.begin());

    // Compute the pairwise distances on our own
    float distances[N*N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < NDIM; k++) {
                sum += (points[i + k*N] - points[j + k*N]) * (points[i + k*N] - points[j + k*N]);
            }
            distances[i*N + j] = sqrt(sum);
        }
    }

    // Compare
    for (int i = 0; i < N*N; i++){
        //EXPECT_TRUE(abs(h_distances[i] - distances[i]) <= 1e-4);
        EXPECT_NEAR(h_distances[i],distances[i], 1e-4);
        EXPECT_TRUE(h_distances[i] >= 0);
    }
}

void test_pairwise_distance_speed(int N, int NDIM) {
    srand (time(NULL));

    // Create some random points in 2 dimensions
    float points[N * NDIM];
    for (int i = 0; i < N*NDIM; i++) {
        points[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    // Construct the thrust vector
    thrust::host_vector<float> h_points(N*NDIM);
    for (int i = 0; i < N*NDIM; i++) 
        h_points[i] = points[i]; 
    thrust::device_vector<float> d_points(N*NDIM);
    thrust::copy(h_points.begin(), h_points.end(), d_points.begin());
    thrust::device_vector<float> d_distances(N*N);

    // Construct the CUBLAS handle
    cublasHandle_t handle;
    CublasSafeCall(cublasCreate(&handle));
    tsnecuda::util::PairwiseDistance(handle, d_distances, d_points, N, NDIM);
}