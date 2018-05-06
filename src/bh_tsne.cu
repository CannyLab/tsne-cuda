// #include "bh_tsne.h"

// #define SOUTHWEST 0u;
// #define NORTHWEST 1u;
// #define SOUTHEAST 2u;
// #define NORTHEAST 3u;

// __global__ void shit::compute_morton_code(const float * __restrict__ pts,
// 									const unsigned int N,
// 									const unsigned int depth,
// 									bounding_box_t bb,
// 									unsigned int * __restrict__ morton_codes) 
// {
// 	unsigned int TID = threadIdx.x + blockIdx.x * blockDim.x;
//     if (TID > N) return;

//     float x = pts[TID];
//     float y = pts[TID + N];

//     unsigned int code = 0u;
//     float center_x, center_y;
//     for (unsigned int level = 0; level < depth; level++) {
//     	code <<= 2;
//     	center_x = (bb.xmin + bb.xmax) / 2.0f;
//     	center_y = (bb.ymin + bb.ymax) / 2.0f;

//     	if (x <= center_x && y <= center_y) {
//     		code += SOUTHWEST;
//     		bb.xmax = center_x;
//     		bb.ymax = center_y;
//     	} else if (x <= center_x && y > center_y) {
//     		code += NORTHWEST;
//     		bb.xmax = center_x;
//     		bb.ymin = center_y;
//     	} else if (x > center_x && y <= center_y) {
//     		code += SOUTHEAST;
//     		bb.xmin = center_x;
//     		bb.ymax = center_y;
//     	} else if (x > center_x && y > center_y) {
//     		code += NORTHEAST;
//     		bb.xmin = center_x;
//     		bb.ymin = center_y;
//     	}
//     }
//     morton_codes[TID] = code;
// }

// //https://github.com/thrust/thrust/blob/master/examples/minmax.cu
// void shit::compute_bounding_box(thrust::device_vector<float> &ys, const unsigned int N, bounding_box_t *bb) {
// 	minmax_unary_op<float>  unary_op;
// 	minmax_binary_op<float> binary_op;

// 	pair_t<float> init = unary_op(ys[0]);
// 	pair_t<float> xlim = thrust::transform_reduce(ys.begin(), ys.begin() + N, unary_op, init, binary_op);

// 	init = unary_op(ys[N]);
// 	pair_t<float> ylim = thrust::transform_reduce(ys.begin() + N, ys.end(), unary_op, init, binary_op);

// 	bb->xmin = xlim.first;
// 	bb->xmax = xlim.second;
// 	bb->ymin = ylim.first;
// 	bb->ymax = ylim.second;
// }

// void shit::sort_points_by_morton_code(thrust::device_vector<float> &ys, 
// 								thrust::device_vector<unsigned int> &morton_codes, 
// 								thrust::device_vector<unsigned int> &indices,
// 								const unsigned int N) {
// 	assert(ys.size() == 2 * N);
// 	assert(morton_codes.size() == N);
// 	assert(indices.size() == N);

// 	bounding_box_t bb;
// 	compute_bounding_box(ys, N, &bb);

// 	const unsigned int BLOCKSIZE = 128;
// 	const unsigned int NBLOCKS = iDivUp(N, BLOCKSIZE);
// 	compute_morton_code<<<NBLOCKS, BLOCKSIZE>>>(thrust::raw_pointer_cast(ys.data()),
// 												N, 10, bb,
// 												thrust::raw_pointer_cast(morton_codes.data()));

// 	// https://stackoverflow.com/questions/6617066/sorting-3-arrays-by-key-in-cuda-using-thrust-perhaps
// 	thrust::counting_iterator<unsigned int> iter(0);
//     thrust::copy(iter, iter + indices.size(), indices.begin());

//     thrust::sort_by_key(morton_codes.begin(), morton_codes.end(), indices.begin());
//     thrust::gather(indices.begin(), indices.end(), ys.begin(), ys.begin());
//     thrust::gather(indices.begin(), indices.end(), ys.begin() + N, ys.begin() + N);
// }


// int main(int argc, char **argv) {
//     const unsigned int N = 100000;
//     // double mypts[] = {-9.99955, -3.83087, -8.29935, 0.297858, 2.02705, -2.0914, -6.2062, 0.885456, 0.299517, 1.84815, -2.03983, -8.1274, -4.74188, 2.18969, 4.87025, -1.35481, -8.20904, 1.99099, 1.64459, -0.15236, 1.83838, -8.56676, 0.234251, -6.16282, 4.52423, -8.32066, -4.05795, 2.33394, -1.47898, 1.58283, 3.05997, -3.71737, -4.11948, -6.72155, -1.70711, -3.30304, 1.2078, 6.89855, -6.70574, 8.17778, 9.35911, 3.78283, 7.53268, -5.52784, 9.33223, -6.37314, 7.98995, 4.78623, 8.03069, -1.22313, 9.23066, -1.76031, 8.13689, 1.54116, 8.72487, 3.97043, 7.83223, 5.79569, 6.19133, 5.45693, 9.90169, 5.60734, 7.15975, 9.84611};
//     // thrust::device_vector<float> pts(2 * N);
//     // for (int i = 0; i < N; i++) {
//         // pts[N - i - 1] = mypts[2 * i];
//         // pts[2 * N - i - 1] = mypts[2 * i + 1];
//     // }
//     auto pts = tsne::util::RandomDeviceUniformZeroOneVector(N * 2);

//     // std::cout << "initial points" << std::endl;
//     // printarray(pts, N, 2);
//     thrust::device_vector<unsigned int> morton_codes(N);
//     thrust::device_vector<unsigned int> indices(N);
//     // std::cout << std::endl << "morton codes" << std::endl;
//     // for (int i = 0; i < N; i++) {
//         // std::cout << morton_codes[i] << " ";
//     // }
//     // std::cout << std::endl << std::endl << "sorted points" << std::endl;
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     printf("Begin sorting with %lu points\n", N);
//     cudaEventRecord(start);
//     sort_points_by_morton_code(pts, morton_codes, indices, N);
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Elapsed time: %f (ms)\n", milliseconds);
//     // printarray(pts, N, 2);
// }

