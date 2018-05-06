// TODO: add copyright

/*
    Compute the sparse pij matrix and index vectors for barnes hut or the dense pij matrix for naive t-SNE.
*/

#include "compute_pij.h"

// BARNES-HUT Compute Pij
// Actual ComputePijKernel is in search_perplexity.cu
// Name of function matches cusparse function call instead of standard style
__global__
void tsnecuda::bh::csr2coo(volatile int * __restrict__ coo_indices, 
                           const int   * __restrict__ pij_row_ptr,
                           const int   * __restrict__ pij_col_ind,
                           const uint32_t num_points,
                           const uint32_t num_nonzero)
{
    register int TID, i, j, start, end;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_nonzero) return;
    start = 0; end = num_points + 1;
    i = (num_points + 1) >> 1;
    while (end - start > 1) {
      j = pij_row_ptr[i];
      end = (j <= TID) ? end : i;
      start = (j > TID) ? start : i;
      i = (start + end) >> 1;
    }
    j = pij_col_ind[TID];
    coo_indices[2*TID] = i;
    coo_indices[2*TID+1] = j;
}

// TODO: Add -1 notification here... and how to deal with it if it happens
// TODO: Maybe think about getting FAISS to return integers (long-term todo)
__global__ void tsnecuda::bh::PostprocessNearestNeighborMatrix(
                                    volatile float * __restrict__ matrix, 
                                    volatile int * __restrict__ neighbor_indices,
                                    const long * __restrict__ long_neighbor_indices,
                                    const uint32_t num_points,
                                    const uint32_t num_near_neighbors)
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_points * num_near_neighbors) return;

    // Set pij to 0 for each of the broken values - Note: this should be handled in the ComputePijKernel now
    // if (matrix[TID] == 1.0f) matrix[TID] = 0.0f;
    neighbor_indices[TID] = (int) long_neighbor_indices[TID];
}

void tsnecuda::bh::ComputePij()

