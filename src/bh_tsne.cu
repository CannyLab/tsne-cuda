#include "bh_tsne.h"

#define SOUTHWEST 0u;
#define NORTHWEST 1u;
#define SOUTHEAST 2u;
#define NORTHEAST 3u;

__global__ void compute_morton_code(const float * __restrict__ pts,
									const unsigned int N,
									const unsigned int depth,
									bounding_box_t bb,
									unsigned int * __restrict__ morton_codes) 
{
	unsigned int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID > N) return;

    float x = pts[TID];
    float y = pts[TID + N];

    unsigned int code = 0u;
    float center_x, center_y;
    for (unsigned int level = 0; level < depth; level++) {
    	code <<= 2;
    	center_x = (bb.xmin + bb.xmax) / 2.0f;
    	center_y = (bb.ymin + bb.ymax) / 2.0f;

    	if (x <= center_x && y <= center_y) {
    		code += SOUTHWEST;
    		bb.xmax = center_x;
    		bb.ymax = center_y;
    	} else if (x <= center_x && y > center_y) {
    		code += NORTHWEST;
    		bb.xmax = center_x;
    		bb.ymin = center_y;
    	} else if (x > center_x && y <= center_y) {
    		code += SOUTHEAST;
    		bb.xmin = center_x;
    		bb.ymax = center_y;
    	} else if (x > center_x && y > center_y) {
    		code += NORTHEAST;
    		bb.xmin = center_x;
    		bb.ymin = center_y;
    	}
    }
    morton_codes[TID] = code;
}

//https://github.com/thrust/thrust/blob/master/examples/minmax.cu
void compute_bounding_box(thrust::device_vector<float> &ys, const unsigned int N, bounding_box_t *bb) {
	minmax_unary_op<float>  unary_op;
	minmax_binary_op<float> binary_op;

	pair_t<float> init = unary_op(ys[0]);
	pair_t<float> xlim = thrust::transform_reduce(ys.begin(), ys.begin() + N, unary_op, init, binary_op);

	init = unary_op(ys[N]);
	pair_t<float> ylim = thrust::transform_reduce(ys.begin() + N, ys.end(), unary_op, init, binary_op);

	bb->xmin = xlim.first;
	bb->xmax = xlim.second;
	bb->ymin = ylim.first;
	bb->ymax = ylim.second;
}

void sort_points_by_morton_code(thrust::device_vector<float> &ys, 
								thrust::device_vector<unsigned int> &morton_codes, 
								thrust::device_vector<unsigned int> &indices,
								const unsigned int N) {
	assert(ys.size() == 2 * N);
	assert(morton_codes.size() == N);
	assert(indices.size() == N);

	bounding_box_t bb;
	compute_bounding_box(ys, N, &bb);

	const unsigned int BLOCKSIZE = 128;
	const unsigned int NBLOCKS = iDivUp(N, BLOCKSIZE);
	compute_morton_code<<<NBLOCKS, BLOCKSIZE>>>(thrust::raw_pointer_cast(ys.data()),
												N, 10, bb,
												thrust::raw_pointer_cast(morton_codes.data()))

	// https://stackoverflow.com/questions/6617066/sorting-3-arrays-by-key-in-cuda-using-thrust-perhaps
	thrust::counting_iterator<unsigned int> iter(0);
    thrust::copy(iter, iter + indices.size(), indices.begin());

    thrust::sort_by_key(morton_codes.begin(), morton_codes.end(), indices.begin());
    thrust::gather(indices.begin(), indices.end(), ys.begin(), ys.begin());
    thrust::gather(indices.begin(), indices.end(), ys.begin() + N, ys.begin() + N);
}



