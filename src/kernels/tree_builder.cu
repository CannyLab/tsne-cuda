// TODO: add copyright

/*
        Builds a quad tree for computing Barnes-Hut approximation of t-SNE repulsive forces.
*/

#include "tree_builder.h"

#ifdef __KEPLER__
#define TREE_THREADS 1024
#define TREE_BLOCKS 2
#else
#define TREE_THREADS 512
#define TREE_BLOCKS 3
#endif

/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(1024, 1)
void tsnecuda::bh::ClearKernel1(volatile int * __restrict__ children, const uint32 num_nodes, const uint32 num_points)
{
    register int k, inc, top, bottom;

    top = 4 * num_nodes;
    bottom = 4 * num_points;
    inc = blockDim.x * gridDim.x;
    k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
    if (k < bottom) k += inc;

    // iterate over all cells assigned to thread
    while (k < top) {
        children[k] = -1;
        k += inc;
    }
}


__global__
__launch_bounds__(TREE_THREADS, TREE_BLOCKS)
void tsnecuda::bh::TreeBuildingKernel(volatile int * __restrict__ errd, 
                                        volatile int * __restrict__ children, 
                                        volatile float * __restrict__ x_pos_device, 
                                        volatile float * __restrict__ y_pos_device,
                                        const uint32 num_nodes,
                                        const uint32 num_points) 
{
    register int i, j, depth, localmaxdepth, skip, inc;
    register float x, y, r;
    register float px, py;
    register float dx, dy;
    register int ch, n, cell, locked, patch;
    register float radius, rootx, rooty;

    // cache root data
    radius = radiusd;
    rootx = x_pos_device[num_nodes];
    rooty = y_pos_device[num_nodes];

    localmaxdepth = 1;
    skip = 1;
    inc = blockDim.x * gridDim.x;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    // iterate over all bodies assigned to thread
    while (i < num_points) {
        if (skip != 0) {
          // new body, so start traversing at root
          skip = 0;
          px = x_pos_device[i];
          py = y_pos_device[i];
          n = num_nodes;
          depth = 1;
          r = radius * 0.5f;
          dx = dy = -r;
          j = 0;
          // determine which child to follow
          if (rootx < px) {j = 1; dx = r;}
          if (rooty < py) {j |= 2; dy = r;}
          x = rootx + dx;
          y = rooty + dy;
        }

        // follow path to leaf cell
        ch = children[n*4+j];
        while (ch >= num_points) {
          n = ch;
          depth++;
          r *= 0.5f;
          dx = dy = -r;
          j = 0;
          // determine which child to follow
          if (x < px) {j = 1; dx = r;}
          if (y < py) {j |= 2; dy = r;}
          x += dx;
          y += dy;
          ch = children[n*4+j];
        }
        if (ch != -2) {  // skip if child pointer is locked and try again later
          locked = n*4+j;
          if (ch == -1) {
            if (-1 == atomicCAS((int *)&children[locked], -1, i)) {  // if null, just insert the new body
              localmaxdepth = max(depth, localmaxdepth);
              i += inc;  // move on to next body
              skip = 1;
            }
          } else {  // there already is a body in this position
            if (ch == atomicCAS((int *)&children[locked], ch, -2)) {  // try to lock
              patch = -1;
              // create new cell(s) and insert the old and new body
              do {
                depth++;

                cell = atomicSub((int *)&bottomd, 1) - 1;
                if (cell <= num_points) {
                  *errd = 1;
                  bottomd = num_nodes;
                }

                if (patch != -1) {
                  children[n*4+j] = cell;
                }
                patch = max(patch, cell);
                j = 0;
                if (x < x_pos_device[ch]) j = 1;
                if (y < y_pos_device[ch]) j |= 2;
                children[cell*4+j] = ch;
                n = cell;
                r *= 0.5f;
                dx = dy = -r;
                j = 0;
                if (x < px) {j = 1; dx = r;}
                if (y < py) {j |= 2; dy = r;}
                x += dx;
                y += dy;
                ch = children[n*4+j];
                // repeat until the two bodies are different children
              } while (ch >= 0 && r > 1e-10); // add radius check because bodies that are very close together can cause this to fail... if points are too close together it will exceed the max depth of the tree 
              children[n*4+j] = i;

              localmaxdepth = max(depth, localmaxdepth);
              i += inc;  // move on to next body
              skip = 2;
            }
          }
        }
        __threadfence();

        if (skip == 2) {
          children[locked] = patch;
        }
    }
    // record maximum tree depth
    atomicMax((int *)&maxdepthd, localmaxdepth);
}


__global__
__launch_bounds__(1024, 1)
void tsnecuda::bh::ClearKernel2(volatile int * __restrict__ cell_starts, volatile float * __restrict__ cell_mass, const uint32 num_nodes)
{
    register int k, inc, bottom;

    bottom = bottomd;
    inc = blockDim.x * gridDim.x;
    k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
    if (k < bottom) k += inc;

    // iterate over all cells assigned to thread
    while (k < num_nodes) {
        cell_mass[k] = -1.0f;
        cell_starts[k] = -1;
        k += inc;
    }
}

void tsnecuda::bh::BuildTree(thrust::device_vector<int> &errd,
                               thrust::device_vector<int> &children,
                               thrust::device_vector<int> &cell_starts,
                               thrust::device_vector<float> &points,
                               const uint32 num_nodes,
                               const uint32 num_points,
                               const uint32 num_blocks)
{
    tsnecuda::bh::ClearKernel1<<<num_blocks, 1024>>>(thrust::raw_pointer_cast(children.data()),
                                                       num_nodes, num_points);
    tsnecuda::bh::TreeBuildingKernel<<<num_blocks * TREE_BLOCKS, TREE_THREADS>>>(
                                                                  thrust::raw_pointer_cast(errd.data()),
                                                                  thrust::raw_pointer_cast(children.data()),
                                                                  thrust::raw_pointer_cast(points.data()),
                                                                  thrust::raw_pointer_cast(points.data() + num_nodes + 1),
                                                                  num_nodes, num_points);
    tsnecuda::bh::ClearKernel2<<<num_blocks, 1024>>>(thrust::raw_pointer_cast(cell_starts.data()),
                                                       thrust::raw_pointer_cast(cell_mass.data()),
                                                       num_nodes);
    gpuErrchk(cudaDeviceSynchronize());
}
