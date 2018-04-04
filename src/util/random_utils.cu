/**
 * @brief Utilities for generating random numbers/elements/vectors
 * 
 * @file random_utils.cu
 * @author David Chan
 * @date 2018-04-04
 */

 #include "util/random_utils.h"

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