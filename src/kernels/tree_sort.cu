/*
        Sort points and cells by morton code.
*/

#include "include/kernels/tree_sort.h"

/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(SORT_THREADS, SORT_BLOCKS)
void tsnecuda::bh::SortKernel(int * __restrict__ cell_sorted, 
                              volatile int * __restrict__ cell_starts, 
                              int * __restrict__ children,
                              const int * __restrict__ cell_counts, 
                              const uint32_t num_nodes,
                              const uint32_t num_points)
{
    register int i, j, k, ch, dec, start, bottom;

    bottom = bottomd;
    dec = blockDim.x * gridDim.x;
    k = num_nodes + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

    // iterate over all cells assigned to thread
    while (k >= bottom) {
        start = cell_starts[k];
        if (start >= 0) {
            j = 0;
            for (i = 0; i < 4; i++) {
                ch = children[k*4+i];
                if (ch >= 0) {
                    if (i != j) {
                        // move children to front (needed later for speed)
                        children[k*4+i] = -1;
                        children[k*4+j] = ch;
                    }
                    j++;
                    if (ch >= num_points) {
                        // child is a cell
                        cell_starts[ch] = start;    // set start ID of child
                        start += cell_counts[ch];    // add #bodies in subtree
                    } else {
                        // child is a body
                        cell_sorted[start] = ch;    // record body in 'sorted' array
                        start++;
                    }
                }
            }
            k -= dec;    // move on to next cell
        }
    }
}

void tsnecuda::bh::SortCells(thrust::device_vector<int> &cell_sorted,
                             thrust::device_vector<int> &cell_starts,
                             thrust::device_vector<int> &children,
                             thrust::device_vector<int> &cell_counts,
                             const uint32_t num_nodes,
                             const uint32_t num_points,
                             const uint32_t num_blocks)
{
    tsnecuda::bh::SortKernel<<<num_blocks * SORT_BLOCKS, SORT_THREADS>>>(
                                thrust::raw_pointer_cast(cell_sorted.data()),
                                thrust::raw_pointer_cast(cell_starts.data()),
                                thrust::raw_pointer_cast(children.data()),
                                thrust::raw_pointer_cast(cell_counts.data()),
                                num_nodes, num_points);
    GpuErrorCheck(cudaDeviceSynchronize());
}
