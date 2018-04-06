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
        float operator()(const int &n) const
        {
            thrust::default_random_engine rng;
            thrust::uniform_real_distribution<float> dist(a, b);
            rng.discard(n);
            return dist(rng);
        }
};


thrust::device_vector<float> random_vector(const unsigned int N) {
    thrust::device_vector<float> vec(N);
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + N;
    thrust::transform(first, last, vec.begin(), prg(-10.0f, 10.0f));
    return vec;
}