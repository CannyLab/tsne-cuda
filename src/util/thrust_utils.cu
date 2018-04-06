#include "util/thrust_utils.h"

// Assumes that vec is an N x N matrix
void zero_diagonal(thrust::device_vector<float> &vec, const unsigned int N) {
	typdef thrust::device_vector<float>::iterator Iterator;
	strided_range<Iterator> diag(vec.begin(), vec.end(), N + 1);
	thrust::copy(diag.begin(), diag.end(), squared_norms.begin());
}