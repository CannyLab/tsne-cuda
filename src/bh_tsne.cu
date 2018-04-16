#include "bh_tsne.h"

//https://github.com/thrust/thrust/blob/master/examples/minmax.cu
void compute_bounding_box(thrust::device_vector<float> &ys, const unsigned int N, float *xmin, float *xmax, float *ymin, float *ymax) {
	minmax_unary_op<float>  unary_op;
	minmax_binary_op<float> binary_op;

	pair_t<float> init = unary_op(ys[0]);
	pair_t<float> xlim = thrust::transform_reduce(ys.begin(), ys.begin() + N, unary_op, init, binary_op);

	init = unary_op(ys[N]);
	pair_t<float> ylim = thrust::transform_reduce(ys.begin() + N, ys.end(), unary_op, init, binary_op);

	*xmin = xlim.first;
	*xmax = xlim.second;
	*ymin = ylim.first;
	*ymax = ylim.second;
}
