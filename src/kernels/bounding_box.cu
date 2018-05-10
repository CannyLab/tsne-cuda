/*
        Computes bounding box for all points for barnes hut. Also resets start, child, mass
*/

#include "include/kernels/bounding_box.h"

/******************************************************************************/
/*** compute center and radius ************************************************/
/******************************************************************************/

__global__
__launch_bounds__(BOUNDING_BOX_THREADS, BOUNDING_BOX_BLOCKS)
void tsnecuda::bh::BoundingBoxKernel(
                       volatile int * __restrict__ cell_starts, 
                       volatile int * __restrict__ children, 
                       volatile float * __restrict__ cell_mass, 
                       volatile float * __restrict__ x_pos_device, 
                       volatile float * __restrict__ y_pos_device, 
                       volatile float * __restrict__ x_max_device, 
                       volatile float * __restrict__ y_max_device, 
                       volatile float * __restrict__ x_min_device, 
                       volatile float * __restrict__ y_min_device,
                       const int num_nodes,
                       const int num_points) 
{
    register int i, j, k, inc;
    register float val, minx, maxx, miny, maxy;
    __shared__ volatile float x_min_shared[BOUNDING_BOX_THREADS], x_max_shared[BOUNDING_BOX_THREADS], y_min_shared[BOUNDING_BOX_THREADS], y_max_shared[BOUNDING_BOX_THREADS];

    // initialize with valid data (in case #bodies < #threads)
    minx = maxx = x_pos_device[0];
    miny = maxy = y_pos_device[0];

    // scan all bodies
    i = threadIdx.x;
    inc = BOUNDING_BOX_THREADS * gridDim.x;
    for (j = i + blockIdx.x * BOUNDING_BOX_THREADS; j < num_points; j += inc) {
        val = x_pos_device[j];
        minx = fminf(minx, val);
        maxx = fmaxf(maxx, val);
        val = y_pos_device[j];
        miny = fminf(miny, val);
        maxy = fmaxf(maxy, val);
    }

    // reduction in shared memory
    x_min_shared[i] = minx;
    x_max_shared[i] = maxx;
    y_min_shared[i] = miny;
    y_max_shared[i] = maxy;

    for (j = BOUNDING_BOX_THREADS / 2; j > 0; j /= 2) {
        __syncthreads();
        if (i < j) {
            k = i + j;
            x_min_shared[i] = minx = fminf(minx, x_min_shared[k]);
            x_max_shared[i] = maxx = fmaxf(maxx, x_max_shared[k]);
            y_min_shared[i] = miny = fminf(miny, y_min_shared[k]);
            y_max_shared[i] = maxy = fmaxf(maxy, y_max_shared[k]);
        }
    }

    // write block result to global memory
    if (i == 0) {
        k = blockIdx.x;
        x_min_device[k] = minx;
        x_max_device[k] = maxx;
        y_min_device[k] = miny;
        y_max_device[k] = maxy;
        __threadfence();

        inc = gridDim.x - 1;
        if (inc == atomicInc(&blkcntd, inc)) {
            // I'm the last block, so combine all block results
            for (j = 0; j <= inc; j++) {
                minx = fminf(minx, x_min_device[j]);
                maxx = fmaxf(maxx, x_max_device[j]);
                miny = fminf(miny, y_min_device[j]);
                maxy = fmaxf(maxy, y_max_device[j]);
            }

            // compute 'radius'
            radiusd = fmaxf(maxx - minx, maxy - miny) * 0.5f + 1e-5f;

            // create root node
            k = num_nodes;
            bottomd = k;

            cell_mass[k] = -1.0f;
            cell_starts[k] = 0;
            x_pos_device[k] = (minx + maxx) * 0.5f;
            y_pos_device[k] = (miny + maxy) * 0.5f;
            k *= 4;
            for (i = 0; i < 4; i++) children[k + i] = -1;

            stepd++;
        }
    }
}

void tsnecuda::bh::ComputeBoundingBox(thrust::device_vector<int> &cell_starts,
                                      thrust::device_vector<int> &children,
                                      thrust::device_vector<float> &cell_mass,
                                      thrust::device_vector<float> &points,
                                      thrust::device_vector<float> &x_max_device,
                                      thrust::device_vector<float> &y_max_device,
                                      thrust::device_vector<float> &x_min_device,
                                      thrust::device_vector<float> &y_min_device,
                                      const int num_nodes,
                                      const int num_points,
                                      const int num_blocks)
{
    tsnecuda::bh::BoundingBoxKernel<<<num_blocks * BOUNDING_BOX_BLOCKS, BOUNDING_BOX_THREADS>>>(
                                                          thrust::raw_pointer_cast(cell_starts.data()),
                                                          thrust::raw_pointer_cast(children.data()),
                                                          thrust::raw_pointer_cast(cell_mass.data()),
                                                          thrust::raw_pointer_cast(points.data()),
                                                          thrust::raw_pointer_cast(points.data() + num_nodes + 1),
                                                          thrust::raw_pointer_cast(x_max_device.data()),
                                                          thrust::raw_pointer_cast(y_max_device.data()),
                                                          thrust::raw_pointer_cast(x_min_device.data()),
                                                          thrust::raw_pointer_cast(y_min_device.data()),
                                                          num_nodes, num_points);
    GpuErrorCheck(cudaDeviceSynchronize());
}
