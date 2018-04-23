/**
 * @brief Implementation of Vanderwaals forces-based T-SNE
 * 
 * @file vanderwaals_tsne.cu
 * @author David Chan
 * @date 2018-04-16
 */

thrust::device_vector<float> VWS::tsne(cublasHandle_t &handle, thrust::device_vector<float> &points, 
                                        const unsigned int N, const unsigned int NDIMS, 
                                        const unsigned int PROJDIM) {

    // First step is to comptue the PIJ matrix (we can do this using the same function as the naive T-SNE)
    // however trying to do that, we might possibly get into trouble with runtime bounds

    // Next step is to perform the particle simulation using simple VWS cutoffs. The VWS cutoffs should be 
    // specified by a bucket distance, which can help us determine the force boundaries.

    // There remains a need to determine exactly what the gradients are for a certain set of points, and if
    // we can efficiently compute the gradients. 

    return nullptr;
}
