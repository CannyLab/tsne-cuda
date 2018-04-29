/*
CUDA BarnesHut v3.1: Simulation of the gravitational forces
in a galactic cluster using the Barnes-Hut n-body algorithm

Copyright (c) 2013, Texas State University-San Marcos. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided that
the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, 
     this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University-San Marcos nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University-San Marcos <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher <burtscher@txstate.edu>
*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include "bh_tsne.h"

#ifdef __KEPLER__

// thread count
#define THREADS1 1024  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 768
#define THREADS4 128
#define THREADS5 1024
#define THREADS6 1024

// block count = factor * #SMs
#define FACTOR1 2
#define FACTOR2 2
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 4  /* must all be resident at the same time */
#define FACTOR5 2
#define FACTOR6 2

#else

// thread count
#define THREADS1 512  /* must be a power of 2 */
#define THREADS2 512
#define THREADS3 128
#define THREADS4 64
#define THREADS5 256
#define THREADS6 1024

// block count = factor * #SMs
#define FACTOR1 3
#define FACTOR2 3
#define FACTOR3 6  /* must all be resident at the same time */
#define FACTOR4 6  /* must all be resident at the same time */
#define FACTOR5 5
#define FACTOR6 1

#endif

#define WARPSIZE 32
#define MAXDEPTH 32

__device__ volatile int stepd, bottomd, maxdepthd;
__device__ unsigned int blkcntd;
__device__ volatile float radiusd;


/******************************************************************************/
/*** initialize memory ********************************************************/
/******************************************************************************/

__global__ void InitializationKernel(int * __restrict errd)
{
  *errd = 0;
  stepd = -1;
  maxdepthd = 1;
  blkcntd = 0;
}


/******************************************************************************/
/*** compute center and radius ************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS1, FACTOR1)
void BoundingBoxKernel(int nnodesd, 
                        int nbodiesd, 
                        volatile int * __restrict startd, 
                        volatile int * __restrict childd, 
                        volatile float * __restrict massd, 
                        volatile float * __restrict posxd, 
                        volatile float * __restrict posyd, 
                        volatile float * __restrict maxxd, 
                        volatile float * __restrict maxyd, 
                        volatile float * __restrict minxd, 
                        volatile float * __restrict minyd) 
{
  register int i, j, k, inc;
  register float val, minx, maxx, miny, maxy;
  __shared__ volatile float sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];

  // scan all bodies
  i = threadIdx.x;
  inc = THREADS1 * gridDim.x;
  for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc) {
    val = posxd[j];
    minx = fminf(minx, val);
    maxx = fmaxf(maxx, val);
    val = posyd[j];
    miny = fminf(miny, val);
    maxy = fmaxf(maxy, val);
  }

  // reduction in shared memory
  sminx[i] = minx;
  smaxx[i] = maxx;
  sminy[i] = miny;
  smaxy[i] = maxy;

  for (j = THREADS1 / 2; j > 0; j /= 2) {
    __syncthreads();
    if (i < j) {
      k = i + j;
      sminx[i] = minx = fminf(minx, sminx[k]);
      smaxx[i] = maxx = fmaxf(maxx, smaxx[k]);
      sminy[i] = miny = fminf(miny, sminy[k]);
      smaxy[i] = maxy = fmaxf(maxy, smaxy[k]);
    }
  }

  // write block result to global memory
  if (i == 0) {
    k = blockIdx.x;
    minxd[k] = minx;
    maxxd[k] = maxx;
    minyd[k] = miny;
    maxyd[k] = maxy;
    __threadfence();

    inc = gridDim.x - 1;
    if (inc == atomicInc(&blkcntd, inc)) {
      // I'm the last block, so combine all block results
      for (j = 0; j <= inc; j++) {
        minx = fminf(minx, minxd[j]);
        maxx = fmaxf(maxx, maxxd[j]);
        miny = fminf(miny, minyd[j]);
        maxy = fmaxf(maxy, maxyd[j]);
      }

      // compute 'radius'
      radiusd = fmaxf(maxx - minx, maxy - miny) * 0.5f;

      // create root node
      k = nnodesd;
      bottomd = k;

      massd[k] = -1.0f;
      startd[k] = 0;
      posxd[k] = (minx + maxx) * 0.5f;
      posyd[k] = (miny + maxy) * 0.5f;
      k *= 4;
      for (i = 0; i < 4; i++) childd[k + i] = -1;

      stepd++;
    }
  }
}


/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(1024, 1)
void ClearKernel1(int nnodesd, int nbodiesd, volatile int * __restrict childd)
{
  register int k, inc, top, bottom;

  top = 4 * nnodesd;
  bottom = 4 * nbodiesd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < top) {
    childd[k] = -1;
    k += inc;
  }
}


__global__
__launch_bounds__(THREADS2, FACTOR2)
void TreeBuildingKernel(int nnodesd, 
                        int nbodiesd, 
                        volatile int * __restrict errd, 
                        volatile int * __restrict childd, 
                        volatile float * __restrict posxd, 
                        volatile float * __restrict posyd) 
{
  register int i, j, depth, localmaxdepth, skip, inc;
  register float x, y, r;
  register float px, py;
  register float dx, dy;
  register int ch, n, cell, locked, patch;
  register float radius, rootx, rooty;

  // cache root data
  radius = radiusd;
  rootx = posxd[nnodesd];
  rooty = posyd[nnodesd];

  localmaxdepth = 1;
  skip = 1;
  inc = blockDim.x * gridDim.x;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < nbodiesd) {
    //   if (TID == 0)
        // printf("\tStarting\n");
    if (skip != 0) {
      // new body, so start traversing at root
      skip = 0;
      px = posxd[i];
      py = posyd[i];
      n = nnodesd;
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
    ch = childd[n*4+j];
    while (ch >= nbodiesd) {
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
      ch = childd[n*4+j];
    }
    if (ch != -2) {  // skip if child pointer is locked and try again later
      locked = n*4+j;
      if (ch == -1) {
        if (-1 == atomicCAS((int *)&childd[locked], -1, i)) {  // if null, just insert the new body
          localmaxdepth = max(depth, localmaxdepth);
          i += inc;  // move on to next body
          skip = 1;
        }
      } else {  // there already is a body in this position
        if (ch == atomicCAS((int *)&childd[locked], ch, -2)) {  // try to lock
          patch = -1;
          // create new cell(s) and insert the old and new body
          do {
            depth++;

            cell = atomicSub((int *)&bottomd, 1) - 1;
            if (cell <= nbodiesd) {
              *errd = 1;
              bottomd = nnodesd;
            }

            if (patch != -1) {
              childd[n*4+j] = cell;
            }
            patch = max(patch, cell);

            j = 0;
            if (x < posxd[ch]) j = 1;
            if (y < posyd[ch]) j |= 2;
            childd[cell*4+j] = ch;

            n = cell;
            r *= 0.5f;
            dx = dy = -r;
            j = 0;
            if (x < px) {j = 1; dx = r;}
            if (y < py) {j |= 2; dy = r;}
            x += dx;
            y += dy;

            ch = childd[n*4+j];
            // repeat until the two bodies are different children
          } while (ch >= 0);
          childd[n*4+j] = i;

          localmaxdepth = max(depth, localmaxdepth);
          i += inc;  // move on to next body
          skip = 2;
        }
      }
    }
    __syncthreads();  // __threadfence();

    if (skip == 2) {
      childd[locked] = patch;
    }
  }
  // record maximum tree depth
  atomicMax((int *)&maxdepthd, localmaxdepth);
}


__global__
__launch_bounds__(1024, 1)
void ClearKernel2(int nnodesd, volatile int * __restrict startd, volatile float * __restrict massd)
{
  register int k, inc, bottom;

  bottom = bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < nnodesd) {
    massd[k] = -1.0f;
    startd[k] = -1;
    k += inc;
  }
}


/******************************************************************************/
/*** compute center of mass ***************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernel(const int nnodesd, 
                            const int nbodiesd, 
                            volatile int * __restrict countd, 
                            const int * __restrict childd, 
                            volatile float * __restrict massd, 
                            volatile float * __restrict posxd, 
                            volatile float * __restrict posyd) 
{
  register int i, j, k, ch, inc, cnt, bottom, flag;
  register float m, cm, px, py;
  __shared__ int child[THREADS3 * 4];
  __shared__ float mass[THREADS3 * 4];

  bottom = bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  register int restart = k;
  for (j = 0; j < 5; j++) {  // wait-free pre-passes
    // iterate over all cells assigned to thread
    while (k <= nnodesd) {
      if (massd[k] < 0.0f) {
        for (i = 0; i < 4; i++) {
          ch = childd[k*4+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch >= nbodiesd) && ((mass[i*THREADS3+threadIdx.x] = massd[ch]) < 0.0f)) {
            break;
          }
        }
        if (i == 4) {
          // all children are ready
          cm = 0.0f;
          px = 0.0f;
          py = 0.0f;
          cnt = 0;
          for (i = 0; i < 4; i++) {
            ch = child[i*THREADS3+threadIdx.x];
            if (ch >= 0) {
              if (ch >= nbodiesd) {  // count bodies (needed later)
                m = mass[i*THREADS3+threadIdx.x];
                cnt += countd[ch];
              } else {
                m = massd[ch];
                cnt++;
              }
              // add child's contribution
              cm += m;
              px += posxd[ch] * m;
              py += posyd[ch] * m;
            }
          }
          countd[k] = cnt;
          m = 1.0f / cm;
          posxd[k] = px * m;
          posyd[k] = py * m;
          __threadfence();  // make sure data are visible before setting mass
          massd[k] = cm;
        }
      }
      k += inc;  // move on to next cell
    }
    k = restart;
  }

  flag = 0;
  j = 0;
  // iterate over all cells assigned to thread
  while (k <= nnodesd) {
    if (massd[k] >= 0.0f) {
      k += inc;
    } else {
      if (j == 0) {
        j = 4;
        for (i = 0; i < 4; i++) {
          ch = childd[k*4+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch < nbodiesd) || ((mass[i*THREADS3+threadIdx.x] = massd[ch]) >= 0.0f)) {
            j--;
          }
        }
      } else {
        j = 4;
        for (i = 0; i < 4; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if ((ch < nbodiesd) || (mass[i*THREADS3+threadIdx.x] >= 0.0f) || ((mass[i*THREADS3+threadIdx.x] = massd[ch]) >= 0.0f)) {
            j--;
          }
        }
      }

      if (j == 0) {
        // all children are ready
        cm = 0.0f;
        px = 0.0f;
        py = 0.0f;
        cnt = 0;
        for (i = 0; i < 4; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if (ch >= 0) {
            if (ch >= nbodiesd) {  // count bodies (needed later)
              m = mass[i*THREADS3+threadIdx.x];
              cnt += countd[ch];
            } else {
              m = massd[ch];
              cnt++;
            }
            // add child's contribution
            cm += m;
            px += posxd[ch] * m;
            py += posyd[ch] * m;
          }
        }
        countd[k] = cnt;
        m = 1.0f / cm;
        posxd[k] = px * m;
        posyd[k] = py * m;
        flag = 1;
      }
    }
    __syncthreads();  
    __threadfence();
    if (flag != 0) {
      massd[k] = cm;
      k += inc;
      flag = 0;
    }
  }
}


/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS4, FACTOR4)
void SortKernel(int nnodesd, int nbodiesd, int * __restrict sortd, int * __restrict countd, volatile int * __restrict startd, int * __restrict childd)
{
  register int i, j, k, ch, dec, start, bottom;

  bottom = bottomd;
  dec = blockDim.x * gridDim.x;
  k = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all cells assigned to thread
  while (k >= bottom) {
    start = startd[k];
    if (start >= 0) {
      j = 0;
      for (i = 0; i < 4; i++) {
        ch = childd[k*4+i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front (needed later for speed)
            childd[k*4+i] = -1;
            childd[k*4+j] = ch;
          }
          j++;
          if (ch >= nbodiesd) {
            // child is a cell
            startd[ch] = start;  // set start ID of child
            start += countd[ch];  // add #bodies in subtree
          } else {
            // child is a body
            sortd[start] = ch;  // record body in 'sorted' array
            start++;
          }
        }
      }
      k -= dec;  // move on to next cell
    }
  }
}


/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernel(int nnodesd, 
                            int nbodiesd, 
                            volatile int * __restrict errd, 
                            float theta, 
                            volatile int * __restrict sortd, 
                            volatile int * __restrict childd, 
                            volatile float * __restrict massd, 
                            volatile float * __restrict posxd, 
                            volatile float * __restrict posyd, 
                            volatile float * __restrict velxd, 
                            volatile float * __restrict velyd,
                            volatile float * __restrict normd) 
{
  register int i, j, k, n, depth, base, sbase, diff, pd, nd;
  register float px, py, vx, vy, dx, dy, normsum, tmp, mult;
  __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
  __shared__ float dq[MAXDEPTH * THREADS5/WARPSIZE];

  if (0 == threadIdx.x) {
    // tmp = radiusd * 2;
    // precompute values that depend only on tree level
    dq[0] = radiusd * theta; //tmp * tmp * itolsqd;
    for (i = 1; i < maxdepthd; i++) {
      dq[i] = dq[i - 1] * 0.5f; // radius is halved with every level of the tree
      // dq[i - 1] += epssqd;
    }
    // dq[i - 1] += epssqd;

    if (maxdepthd > MAXDEPTH) {
      *errd = maxdepthd;
    }
  }
  __syncthreads();

  if (maxdepthd <= MAXDEPTH) {
    // figure out first thread in each warp (lane 0)
    base = threadIdx.x / WARPSIZE;
    sbase = base * WARPSIZE;
    j = base * MAXDEPTH;

    diff = threadIdx.x - sbase;
    // make multiple copies to avoid index calculations later
    if (diff < MAXDEPTH) {
      dq[diff+j] = dq[diff];
    }
    __syncthreads();
    __threadfence_block();

    // iterate over all bodies assigned to thread
    for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd; k += blockDim.x * gridDim.x) {
      i = sortd[k];  // get permuted/sorted index
      // cache position info
      px = posxd[i];
      py = posyd[i];

      vx = 0.0f;
      vy = 0.0f;
      normsum = 0.0f;

      // initialize iteration stack, i.e., push root node onto stack
      depth = j;
      if (sbase == threadIdx.x) {
        pos[j] = 0;
        node[j] = nnodesd * 4;
      }

      do {
        // stack is not empty
        pd = pos[depth];
        nd = node[depth];
        while (pd < 4) {
          // node on top of stack has more children to process
          n = childd[nd + pd];  // load child pointer
          pd++;

          if (n >= 0) {
            dx = posxd[n] - px;
            dy = posyd[n] - py;
            tmp = dx*dx + dy*dy; // distance squared
            // tmp = dx*dx + (dy*dy + epssqd) (why softening?)
            if ((n < nbodiesd) || __all(tmp >= dq[depth])) {  // check if all threads agree that cell is far enough away (or is a body)
            //   tmp = rsqrtf(tmp);  // compute distance
              // from sptree.cpp
              tmp = 1 / (1 + tmp);
              mult = massd[n] * tmp;
              normsum += mult;
              mult *= tmp;
              vx += dx * mult;
              vy += dy * mult;
            } else {
              // push cell onto stack
              if (sbase == threadIdx.x) {  // maybe don't push and inc if last child
                pos[depth] = pd;
                node[depth] = nd;
              }
              depth++;
              pd = 0;
              nd = n * 4;
            }
          } else {
            pd = 4;  // early out because all remaining children are also zero
          }
        }
        depth--;  // done with this level
      } while (depth >= j);

      if (stepd >= 0) {
        // update velocity
        // TODO: This is probably wrongish and depends on what I do in the attractive force calculation
        velxd[i] += vx;
        velyd[i] += vy;
        normd[i] = normsum;
      }
    }
  }
}


/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/
// Edited to add momentum, repulsive, attr forces, etc.
__global__
__launch_bounds__(THREADS6, FACTOR6)
void IntegrationKernel(int N,
                        int nnodes,
                        float eta,
                        float norm,
                        float momentum,
                        volatile float * __restrict pts, // (nnodes + 1) x 2
                        volatile float * __restrict attr_forces, // (N x 2)
                        volatile float * __restrict rep_forces, // (nnodes + 1) x 2
                        volatile float * __restrict old_forces) // (N x 2)
{
  register int i, inc;
  register float tmpx, tmpy;

  // iterate over all bodies assigned to thread
  // TODO: fix momentum at step 0
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += inc) {
      tmpx = 4.0f * (attr_forces[i] + (rep_forces[i] / norm));
      tmpy = 4.0f * (attr_forces[i + N] + (rep_forces[nnodes + 1 + i] / norm));
      tmpx = momentum * tmpx + (1 - momentum) * old_forces[i];
      tmpy = momentum * tmpy + (1 - momentum) * old_forces[i + N];
      pts[i] -= eta * tmpx;
      pts[i + nnodes + 1] -= eta * tmpy;
      old_forces[i] = tmpx;
      old_forces[i + N] = tmpy;
   }
}


/******************************************************************************/
/*** compute attractive force *************************************************/
/******************************************************************************/
__global__
void computePijxQij(int N, 
                    int nnz, 
                    int nnodes,
                    volatile float * __restrict pij,
                    volatile int   * __restrict pijRowPtr,
                    volatile int   * __restrict pijColInd,
                    volatile float * __restrict forceProd,
                    volatile float * __restrict pts)
{
    register int TID, i, j, start, end;
    register float ix, iy, jx, jy, dx, dy, tmp;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= nnz) return;
    start = 0; end = N + 1;
    i = (N + 1) >> 1;
    while (end - start > 1) {
      j = pijRowPtr[i];
      end = (j <= TID) ? end : i;
      start = (j > TID) ? start : i;
      // if (j > TID)
          // end = i;
      // else
          // start = i;
      i = (start + end) >> 1;
    }
    // if (!(pijRowPtr[i] <= TID && pijRowPtr[i + 1] > TID))
        // printf("something's wrong!\n");
    
    j = pijColInd[TID - i];
    
    ix = pts[i]; iy = pts[nnodes + 1 + i];
    jx = pts[j]; jy = pts[nnodes + 1 + j];
    dx = ix - jx;
    dy = iy - jy;
    tmp = (1 + dx*dx + dy*dy);
    forceProd[TID] = pij[TID] / tmp;
}

// computes unnormalized attractive forces
void computeAttrForce(int N,
                        int nnz,
                        int nnodes,
                        cusparseHandle_t &handle,
                        cusparseMatDescr_t &descr,
                        thrust::device_vector<float> &sparsePij,
                        thrust::device_vector<int>   &pijRowPtr, // (N + 1)-D vector, should be constant L
                        thrust::device_vector<int>   &pijColInd, // NxL matrix (same shape as sparsePij)
                        thrust::device_vector<float> &forceProd, // NxL matrix
                        thrust::device_vector<float> &pts,       // (nnodes + 1) x 2 matrix
                        thrust::device_vector<float> &forces,    // N x 2 matrix
                        thrust::device_vector<float> &ones)      // N x 2 matrix of ones
{
    const int BLOCKSIZE = 128;
    const int NBLOCKS = iDivUp(nnz, BLOCKSIZE);
    computePijxQij<<<NBLOCKS, BLOCKSIZE>>>(N, nnz, nnodes,
                                            thrust::raw_pointer_cast(sparsePij.data()),
                                            thrust::raw_pointer_cast(pijRowPtr.data()),
                                            thrust::raw_pointer_cast(pijColInd.data()),
                                            thrust::raw_pointer_cast(forceProd.data()),
                                            thrust::raw_pointer_cast(pts.data()));
    gpuErrchk(cudaDeviceSynchronize());
    // compute forces_i = sum_j pij*qij*normalization*yi
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseSafeCall(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            N, 2, N, nnz, &alpha, descr,
                            thrust::raw_pointer_cast(forceProd.data()),
                            thrust::raw_pointer_cast(pijRowPtr.data()),
                            thrust::raw_pointer_cast(pijColInd.data()),
                            thrust::raw_pointer_cast(ones.data()),
                            N, &beta, thrust::raw_pointer_cast(forces.data()),
                            N));
    gpuErrchk(cudaDeviceSynchronize());
    thrust::transform(forces.begin(), forces.begin() + N, pts.begin(), forces.begin(), thrust::multiplies<float>());
    thrust::transform(forces.begin() + N, forces.end(), pts.begin() + nnodes + 1, forces.begin() + N, thrust::multiplies<float>());

    // compute forces_i = forces_i - sum_j pij*qij*normalization*yj
    alpha = -1.0f;
    beta = 1.0f;
    cusparseSafeCall(cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            N, 2, N, nnz, &alpha, descr,
                            thrust::raw_pointer_cast(forceProd.data()),
                            thrust::raw_pointer_cast(pijRowPtr.data()),
                            thrust::raw_pointer_cast(pijColInd.data()),
                            thrust::raw_pointer_cast(pts.data()),
                            nnodes + 1, &beta, thrust::raw_pointer_cast(forces.data()),
                            N));
    gpuErrchk(cudaDeviceSynchronize());
    

}

//TODO: Remove NDIMS argument
void compute_pij(cublasHandle_t &handle, 
                    thrust::device_vector<float> &pij,  
                    const thrust::device_vector<float> &knn_distances, 
                    const thrust::device_vector<float> &sigma,
                    const unsigned int N, 
                    const unsigned int K,
                    const unsigned int NDIMS) 
{
    // Square the distances
    Math::square(knn_distances, pij);

    // Square the sigmas
    // TODO: This allocates memory (we may want to fix it....)
    thrust::device_vector<float> sigma_squared(sigma.size());
    Math::square(sigma, sigma_squared);

    // PIJ is KxN. :)
    Broadcast::broadcast_matrix_vector(pij, sigma_squared, K, N, thrust::divides<float>(), 1, -2.0f); // Divide by -2sigma
    thrust::transform(pij.begin(), pij.end(), pij.begin(), func_exp()); //Exponentiate
}

void normalize_pij(cublasHandle_t &handle, 
                    thrust::device_vector<float> &pij,
                    const unsigned int N,
                    const unsigned int K) 
{
    // Reduce::reduce_sum over cols
    auto sums = Reduce::reduce_sum(handle, pij, K, N, 0);
    // divide column by resulting vector
    Broadcast::broadcast_matrix_vector(pij, sums, K, N, thrust::divides<float>(), 1, 1.0f);
}

// TODO: Add -1 notification here...
__global__ void postprocess_matrix(float* matrix, 
                                    long* long_indices,
                                    int* indices,
                                    unsigned int N_POINTS,
                                    unsigned int K) 
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= N_POINTS*K) return;

    // Set pij to 0 for each of the broken values
    if (matrix[TID] == 1.0f) matrix[TID] = 0.0f;
    indices[TID] = (int) long_indices[TID];
    return;
}

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    saxpy_functor(float _a) : a(_a) {}
    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            return a * x + y;
        }
};

struct func_kl {
  __host__ __device__ float operator()(const float &x, const float &y) const { 
      return x == 0.0f ? 0.0f : x * (log(x) - log(y));
  }
};

struct func_entropy_kernel {
  __host__ __device__ float operator()(const float &x) const { float val = x*log2(x); return (val != val || isinf(val)) ? 0 : val; }
};

struct func_pow2 {
  __host__ __device__ float operator()(const float &x) const { return pow(2,x); }
};

__global__ void upper_lower_assign_bh(float * __restrict__ sigmas,
                                      float * __restrict__ lower_bound,
                                      float * __restrict__ upper_bound,
                                      const float * __restrict__ perplexity,
                                      const float target_perplexity,
                                      const unsigned int N)
{
  int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID > N) return;

  if (perplexity[TID] > target_perplexity)
    upper_bound[TID] = sigmas[TID];
  else
    lower_bound[TID] = sigmas[TID];
  sigmas[TID] = (upper_bound[TID] + lower_bound[TID])/2.0f;
}

void thrust_search_perplexity(cublasHandle_t &handle,
                                thrust::device_vector<float> &sigmas,
                                thrust::device_vector<float> &lower_bound,
                                thrust::device_vector<float> &upper_bound,
                                thrust::device_vector<float> &perplexity,
                                const thrust::device_vector<float> &pij,
                                const float target_perplexity,
                                const unsigned int N,
                                const unsigned int K)
{
//   std::cout << "pij:" << std::endl;
//   printarray(pij, N, K);
//   std::cout << std::endl;
  thrust::device_vector<float> entropy_(pij.size());
  thrust::transform(pij.begin(), pij.end(), entropy_.begin(), func_entropy_kernel());

//   std::cout << "entropy:" << std::endl;
//   printarray(entropy_, N, K);
//   std::cout << std::endl;

  auto neg_entropy = Reduce::reduce_alpha(handle, entropy_, K, N, -1.0f, 0);

//   std::cout << "neg_entropy:" << std::endl;
//   printarray(neg_entropy, 1, N);
//   std::cout << std::endl;
  
  thrust::transform(neg_entropy.begin(), neg_entropy.end(), perplexity.begin(), func_pow2());
 
//   std::cout << "perplexity:" << std::endl;
//   printarray(perplexity, 1, N);
//   std::cout << std::endl;

  const unsigned int BLOCKSIZE = 128;
  const unsigned int NBLOCKS = iDivUp(N, BLOCKSIZE);
  upper_lower_assign_bh<<<NBLOCKS,BLOCKSIZE>>>(thrust::raw_pointer_cast(sigmas.data()),
                thrust::raw_pointer_cast(lower_bound.data()),
                thrust::raw_pointer_cast(upper_bound.data()),
                thrust::raw_pointer_cast(perplexity.data()),
                target_perplexity,
                N);
//   std::cout << "sigmas" << std::endl;
//   printarray(sigmas, 1, N);
//   std::cout << std::endl;

}

thrust::device_vector<float> search_perplexity(cublasHandle_t &handle,
                                                  thrust::device_vector<float> &knn_distances,
                                                  const float perplexity_target,
                                                  const float eps,
                                                  const unsigned int N,
                                                  const unsigned int K) 
{
    thrust::device_vector<float> sigmas(N, 500.0f);
    thrust::device_vector<float> best_sigmas(N);
    thrust::device_vector<float> perplexity(N);
    thrust::device_vector<float> lbs(N, 0.0f);
    thrust::device_vector<float> ubs(N, 1000.0f);
    thrust::device_vector<float> pij(N*K);

    compute_pij(handle, pij, knn_distances, sigmas, N, K, 0);
    normalize_pij(handle, pij, N, K);
    float best_perplexity = 1000.0f;
    float perplexity_diff = 50.0f;
    int iters = 0;
    while (perplexity_diff > eps) {
        thrust_search_perplexity(handle, sigmas, lbs, ubs, perplexity, pij, perplexity_target, N, K);
        perplexity_diff = abs(thrust::reduce(perplexity.begin(), perplexity.end())/((float) N) - perplexity_target);
        if (perplexity_diff < best_perplexity){
            best_perplexity = perplexity_diff;
            printf("!! Best perplexity found in %d iterations: %0.5f\n", iters, perplexity_diff);
            thrust::copy(sigmas.begin(), sigmas.end(), best_sigmas.begin());
        }
        compute_pij(handle, pij, knn_distances, sigmas, N, K, 0);
        normalize_pij(handle, pij, N, K);
        iters++;
    } // Close perplexity search

    return pij;
}


thrust::device_vector<float> BHTSNE::tsne(cublasHandle_t &dense_handle, 
                                          cusparseHandle_t &sparse_handle,
                                            float* points, 
                                            unsigned int N_POINTS, 
                                            unsigned int N_DIMS, 
                                            unsigned int PROJDIM, 
                                            float perplexity, 
                                            float learning_rate, 
                                            float early_ex, 
                                            unsigned int n_iter, 
                                            unsigned int n_iter_np, 
                                            float min_g_norm)
{

    cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
    #ifdef __KEPLER__
    cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferEqual);
    #else
    cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
    #endif
    cudaFuncSetCacheConfig(IntegrationKernel, cudaFuncCachePreferL1);

    //TODO: Make this an argument
    const unsigned int K = 1023 < N_POINTS ? 1023 : N_POINTS - 1; 
    float *knn_distances = new float[N_POINTS*K];
    memset(knn_distances, 0, N_POINTS * K * sizeof(float));
    long *knn_indices = new long[N_POINTS*K]; // Allocate memory for the indices on the CPU

    // Compute the KNNs and distances
    Distance::knn(points, knn_indices, knn_distances, N_DIMS, N_POINTS, K);

    // Copy the distances to the GPU
    thrust::device_vector<float> d_knn_distances(N_POINTS*K);
    thrust::copy(knn_distances, knn_distances + N_POINTS*K, d_knn_distances.begin());

    // printarray<float>(d_knn_distances, N_POINTS, K);

    // Normalize the knn distances - this may not be necessary
    // TODO: Need to filter out zeros from distances
    // TODO: Some point is fucked up
    // TODO: nprobe / nlist issue? Check if there are -1s floating around...
    Math::max_norm(d_knn_distances); // Here, the extra 0s floating around won't matter

    // printarray<float>(d_knn_distances, N_POINTS, K);
    
    // Compute the pij of the KNN distribution

    //TODO: Make these arguments
    thrust::device_vector<float> d_pij = search_perplexity(dense_handle, d_knn_distances, perplexity, 1e-4, N_POINTS, K);

    // thrust::device_vector<float> sigmas(N_POINTS, 0.3);
    // thrust::device_vector<float> d_pij(N_POINTS*K);
    // compute_pij(dense_handle, d_pij, d_knn_distances, sigmas, N_POINTS, K, N_DIMS);

    // std::cout << std::endl;
    // printarray<float>(d_pij, N_POINTS, K);

    // Clean up the d_knn_distances matrix
    d_knn_distances.clear();
    d_knn_distances.shrink_to_fit();
    
    // Allocate memory for the indices
    thrust::device_vector<long> d_knn_indices_long(N_POINTS*K);
    thrust::device_vector<int> d_knn_indices(N_POINTS*K);
    thrust::copy(knn_indices, knn_indices + N_POINTS*K, d_knn_indices_long.begin());

    // std::cout << std::endl;
    // printarray<long>(d_knn_indices_long, N_POINTS, K);

    // Post-process the pij matrix-indives to remove zero elements/check for -1 elements
    const int NBLOCKS_PP = iDivUp(N_POINTS*K, 128);
    postprocess_matrix<<< NBLOCKS_PP, 128 >>>(thrust::raw_pointer_cast(d_pij.data()), 
                                              thrust::raw_pointer_cast(d_knn_indices_long.data()), 
                                              thrust::raw_pointer_cast(d_knn_indices.data()),  N_POINTS, K);
    cudaDeviceSynchronize();

    // Clean up extra memory
    d_knn_indices_long.clear();
    d_knn_indices_long.shrink_to_fit();
    delete[] knn_distances;
    delete[] knn_indices;

    // Normalize the pij matrix, we do this after post-processing to avoid issues
    // in the distribution caused by exponentiation.
    // normalize_pij(dense_handle, d_pij, N_POINTS, K);

    // Construct some additional descriptors for sparse matrix multiplication (for the symmetrization)
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    // Symmetrize d_pij
    thrust::device_vector<float> sparsePij; // Device
    thrust::device_vector<int> pijRowPtr; // Device
    thrust::device_vector<int> pijColInd; // Device
    int sym_nnz;
    Sparse::sym_mat_gpu(d_pij, d_knn_indices, sparsePij, pijColInd, pijRowPtr, &sym_nnz, N_POINTS, K);

    // Clear some old memory
    d_knn_indices.clear();
    d_knn_indices.shrink_to_fit();
    d_pij.clear();
    d_pij.shrink_to_fit();  

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.warpSize != WARPSIZE) {
      fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
      exit(-1);
    }

    int blocks = deviceProp.multiProcessorCount;

    int nnodes = N_POINTS * 2;
    if (nnodes < 1024*blocks) nnodes = 1024*blocks;
    while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
    nnodes--;

    thrust::device_vector<float> forceProd(sparsePij.size());
    thrust::device_vector<float> pts = Random::random_vector((nnodes + 1) * 2); //TODO: Rename this function
    thrust::device_vector<float> rep_forces((nnodes + 1) * 2, 0);
    thrust::device_vector<float> attr_forces(N_POINTS * 2, 0);
    thrust::device_vector<float> old_forces(N_POINTS * 2, 0); // for momentum

    thrust::device_vector<int> errl(1);
    thrust::device_vector<int> startl(nnodes + 1);
    thrust::device_vector<int> childl((nnodes + 1) * 4);
    thrust::device_vector<float> massl(nnodes + 1, 1.0); // TODO: probably don't need massl
    thrust::device_vector<int> countl(nnodes + 1);
    thrust::device_vector<int> sortl(nnodes + 1);
    thrust::device_vector<float> norml(nnodes + 1);

    thrust::device_vector<float> maxxl(blocks * FACTOR1);
    thrust::device_vector<float> maxyl(blocks * FACTOR1);
    thrust::device_vector<float> minxl(blocks * FACTOR1);
    thrust::device_vector<float> minyl(blocks * FACTOR1);
    
    thrust::device_vector<float> ones(N_POINTS * 2, 1); // This is for reduce summing, etc.

    float eta = learning_rate * early_ex;
    float momentum = 0.8f;
    float norm;
    
    // These variables currently govern the tolerance (whether it recurses on a cell)
    float theta = 0.5f;
    InitializationKernel<<<1, 1>>>(thrust::raw_pointer_cast(errl.data()));
    gpuErrchk(cudaDeviceSynchronize());
    
    std::ofstream dump_file;
    dump_file.open("dump_ys.txt");
    float host_ys[(nnodes + 1) * 2];
    dump_file << N_POINTS << " " << 2 << std::endl;
    
    for (int step = 0; step < n_iter; step++) {
        thrust::fill(rep_forces.begin(), rep_forces.end(), 0);
        
        // Repulsive force with Barnes Hut
        BoundingBoxKernel<<<blocks * FACTOR1, THREADS1>>>(nnodes, 
                                                          N_POINTS, 
                                                          thrust::raw_pointer_cast(startl.data()), 
                                                          thrust::raw_pointer_cast(childl.data()), 
                                                          thrust::raw_pointer_cast(massl.data()), 
                                                          thrust::raw_pointer_cast(pts.data()), 
                                                          thrust::raw_pointer_cast(pts.data() + nnodes + 1), 
                                                          thrust::raw_pointer_cast(maxxl.data()), 
                                                          thrust::raw_pointer_cast(maxyl.data()), 
                                                          thrust::raw_pointer_cast(minxl.data()), 
                                                          thrust::raw_pointer_cast(minyl.data()));

        gpuErrchk(cudaDeviceSynchronize());

        ClearKernel1<<<blocks * 1, 1024>>>(nnodes, N_POINTS, thrust::raw_pointer_cast(childl.data()));
        TreeBuildingKernel<<<blocks * FACTOR2, THREADS2>>>(nnodes, N_POINTS, thrust::raw_pointer_cast(errl.data()), 
                                                                             thrust::raw_pointer_cast(childl.data()), 
                                                                             thrust::raw_pointer_cast(pts.data()), 
                                                                             thrust::raw_pointer_cast(pts.data() + nnodes + 1));
        ClearKernel2<<<blocks * 1, 1024>>>(nnodes, thrust::raw_pointer_cast(startl.data()), thrust::raw_pointer_cast(massl.data()));
        gpuErrchk(cudaDeviceSynchronize());
        
        SummarizationKernel<<<blocks * FACTOR3, THREADS3>>>(nnodes, N_POINTS, thrust::raw_pointer_cast(countl.data()), 
                                                                                      thrust::raw_pointer_cast(childl.data()), 
                                                                                      thrust::raw_pointer_cast(massl.data()),
                                                                                      thrust::raw_pointer_cast(pts.data()),
                                                                                      thrust::raw_pointer_cast(pts.data() + nnodes + 1));
        gpuErrchk(cudaDeviceSynchronize());
        
        SortKernel<<<blocks * FACTOR4, THREADS4>>>(nnodes, N_POINTS, thrust::raw_pointer_cast(sortl.data()), 
                                                                     thrust::raw_pointer_cast(countl.data()), 
                                                                     thrust::raw_pointer_cast(startl.data()), 
                                                                     thrust::raw_pointer_cast(childl.data()));
        gpuErrchk(cudaDeviceSynchronize());
        
        ForceCalculationKernel<<<blocks * FACTOR5, THREADS5>>>(nnodes, N_POINTS, thrust::raw_pointer_cast(errl.data()), 
                                                                    theta,
                                                                    thrust::raw_pointer_cast(sortl.data()), 
                                                                    thrust::raw_pointer_cast(childl.data()), 
                                                                    thrust::raw_pointer_cast(massl.data()), 
                                                                    thrust::raw_pointer_cast(pts.data()),
                                                                    thrust::raw_pointer_cast(pts.data() + nnodes + 1),
                                                                    thrust::raw_pointer_cast(rep_forces.data()),
                                                                    thrust::raw_pointer_cast(rep_forces.data() + nnodes + 1),
                                                                    thrust::raw_pointer_cast(norml.data()));
        gpuErrchk(cudaDeviceSynchronize());

        // compute attractive forces
        computeAttrForce(N_POINTS, sparsePij.size(), nnodes, sparse_handle, descr, sparsePij, pijRowPtr, pijColInd, forceProd, pts, attr_forces, ones);
        gpuErrchk(cudaDeviceSynchronize());
        
        norm = thrust::reduce(norml.begin(), norml.begin() + N_POINTS, 0.0f, thrust::plus<float>());
        if (step % 10 == 0)
            std::cout << "Step: " << step << ", Norm: " << norm << std::endl;
      
        IntegrationKernel<<<blocks * FACTOR6, THREADS6>>>(N_POINTS, nnodes, eta, norm, momentum, 
                                                                    thrust::raw_pointer_cast(pts.data()),
                                                                    thrust::raw_pointer_cast(attr_forces.data()),
                                                                    thrust::raw_pointer_cast(rep_forces.data()),
                                                                    thrust::raw_pointer_cast(old_forces.data()));


        if (step == 250) {eta /= early_ex;}
        // std::cout << "ATTR FORCES:" << std::endl;
        // for (int i = 0; i < 128; i++) {
            // std::cout << attr_forces[i] << ", " << attr_forces[i + N_POINTS] << std::endl;
        // }
        // std::cout << std::endl;
        // std::cout << "REP FORCES:" << std::endl;
        // for (int i = 0; i < N_POINTS; i++) {
            // std::cout << rep_forces[i] / norm << ", " << rep_forces[i + nnodes + 1] / norm << std::endl;
        // }
        // Add resulting force vector to positions w/ normalization, mul by 4 and learning rate
        // thrust::transform(rep_forces.begin(), rep_forces.begin() + N_POINTS, attr_forces.begin(), attr_forces.begin(), saxpy_functor(1 / norm));
        // thrust::transform(rep_forces.begin() + nnodes + 1, rep_forces.begin() + nnodes + 1 + N_POINTS, attr_forces.begin() + N_POINTS, attr_forces.begin() + N_POINTS, saxpy_functor(1 / norm));

        // thrust::transform(attr_forces.begin(), attr_forces.begin() + N_POINTS, pts.begin(), pts.begin(), saxpy_functor(-eta * 4.0f));
        // thrust::transform(attr_forces.begin() + N_POINTS, attr_forces.end(), pts.begin() + nnodes + 1, pts.begin() + nnodes + 1, saxpy_functor(-eta * 4.0f));
        // for (int i = 0; i < N_POINTS; i++) {
            // std::cout << attr_forces[i] << ", " << attr_forces[i + N_POINTS] << std::endl;
        // }
        
        thrust::copy(pts.begin(), pts.end(), host_ys);
        for (int i = 0; i < N_POINTS; i++) {
            dump_file << host_ys[i] << " " << host_ys[i + nnodes + 1] << std::endl;
        }
        // exit(1);
        // Done (check progress, etc.)
        // if (step >= 20)
            // exit(1);
    }
    dump_file.close();
    std::cout << "Fin." << std::endl;

    return pts;
}

/******************************************************************************/

// static void CudaTest(const char *msg)
// {
//   cudaError_t e;

//   cudaThreadSynchronize();
//   if (cudaSuccess != (e = cudaGetLastError())) {
//     fprintf(stderr, "%s: %d\n", msg, e);
//     fprintf(stderr, "%s\n", cudaGetErrorString(e));
//     exit(-1);
//   }
// }


/******************************************************************************/

// random number generator

#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

// static int A = 1;
// static int B = 0;
// static int randx = 1;
// static int lastrand;


// static void drndset(int seed)
// {
//    A = 1;
//    B = 0;
//    randx = (A * seed + B) & MASK;
//    A = (MULT * A) & MASK;
//    B = (MULT * B + ADD) & MASK;
// }


// static double drnd()
// {
//    lastrand = randx;
//    randx = (A * randx + B) & MASK;
//    return (double)lastrand / TWOTO31;
// }


/******************************************************************************/

// int main(int argc, char *argv[])
// {
//   register int i, run, blocks;
//   int nnodes, nbodies, step, timesteps;
//   register double runtime;
//   int error;
//   register float dtime, dthf, epssq, itolsq;
//   float time, timing[7];
//   cudaEvent_t start, stop;
//   float *mass, *posx, *posy, *velx, *vely;

//   int *errl, *sortl, *childl, *countl, *startl;
//   float *massl;
//   float *posxl, *posyl;
//   float *velxl, *velyl;
//   float *maxxl, *maxyl;
//   float *minxl, *minyl;
//   float *norml;
//   register double rsc, vsc, r, v, x, y, sq, scale;

//   // perform some checks

//   printf("CUDA BarnesHut v3.1 ");
// #ifdef __KEPLER__
//   printf("[Kepler]\n");
// #else
//   printf("[Fermi]\n");
// #endif
//   printf("Copyright (c) 2013, Texas State University-San Marcos. All rights reserved.\n");
//   fflush(stdout);
//   if (argc != 4) {
//     fprintf(stderr, "\n");
//     fprintf(stderr, "arguments: number_of_bodies number_of_timesteps device\n");
//     exit(-1);
//   }

//   int deviceCount;
//   cudaGetDeviceCount(&deviceCount);
//   if (deviceCount == 0) {
//     fprintf(stderr, "There is no device supporting CUDA\n");
//     exit(-1);
//   }

//   const int dev = atoi(argv[3]);
//   if ((dev < 0) || (deviceCount <= dev)) {
//     fprintf(stderr, "There is no device %d\n", dev);
//     exit(-1);
//   }
//   cudaSetDevice(dev);

//   cudaDeviceProp deviceProp;
//   cudaGetDeviceProperties(&deviceProp, dev);
//   if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
//     fprintf(stderr, "There is no CUDA capable device\n");
//     exit(-1);
//   }
//   if (deviceProp.major < 2) {
//     fprintf(stderr, "Need at least compute capability 2.0\n");
//     exit(-1);
//   }
//   if (deviceProp.warpSize != WARPSIZE) {
//     fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
//     exit(-1);
//   }

//   blocks = deviceProp.multiProcessorCount;
// //  fprintf(stderr, "blocks = %d\n", blocks);

//   if ((WARPSIZE <= 0) || (WARPSIZE & (WARPSIZE-1) != 0)) {
//     fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
//     exit(-1);
//   }
//   if (MAXDEPTH > WARPSIZE) {
//     fprintf(stderr, "MAXDEPTH must be less than or equal to WARPSIZE\n");
//     exit(-1);
//   }
//   if ((THREADS1 <= 0) || (THREADS1 & (THREADS1-1) != 0)) {
//     fprintf(stderr, "THREADS1 must be greater than zero and a power of two\n");
//     exit(-1);
//   }

//   // set L1/shared memory configuration
//   cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
//   cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
//   cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
//   cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
//   cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
//   cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
// #ifdef __KEPLER__
//   cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferEqual);
// #else
//   cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
// #endif
//   cudaFuncSetCacheConfig(IntegrationKernel, cudaFuncCachePreferL1);

//   cudaGetLastError();  // reset error value
//   for (run = 0; run < 3; run++) {
//     for (i = 0; i < 7; i++) timing[i] = 0.0f;

//     nbodies = atoi(argv[1]);
//     if (nbodies < 1) {
//       fprintf(stderr, "nbodies is too small: %d\n", nbodies);
//       exit(-1);
//     }
//     if (nbodies > (1 << 30)) {
//       fprintf(stderr, "nbodies is too large: %d\n", nbodies);
//       exit(-1);
//     }
//     nnodes = nbodies * 2;
//     if (nnodes < 1024*blocks) nnodes = 1024*blocks;
//     while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
//     nnodes--;

//     timesteps = atoi(argv[2]);
//     dtime = 0.025;  dthf = dtime * 0.5f;
//     epssq = 0.05 * 0.05;
//     itolsq = 1.0f / (0.5 * 0.5);

//     // allocate memory

//     if (run == 0) {
//       printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

//       mass = (float *)malloc(sizeof(float) * nbodies);
//       if (mass == NULL) {fprintf(stderr, "cannot allocate mass\n");  exit(-1);}
//       posx = (float *)malloc(sizeof(float) * nbodies);
//       if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
//       posy = (float *)malloc(sizeof(float) * nbodies);
//       if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
//       velx = (float *)malloc(sizeof(float) * nbodies);
//       if (velx == NULL) {fprintf(stderr, "cannot allocate velx\n");  exit(-1);}
//       vely = (float *)malloc(sizeof(float) * nbodies);
//       if (vely == NULL) {fprintf(stderr, "cannot allocate vely\n");  exit(-1);}

//       if (cudaSuccess != cudaMalloc((void **)&errl, sizeof(int))) fprintf(stderr, "could not allocate errd\n");  CudaTest("couldn't allocate errd");
//       if (cudaSuccess != cudaMalloc((void **)&childl, sizeof(int) * (nnodes+1) * 4)) fprintf(stderr, "could not allocate childd\n");  CudaTest("couldn't allocate childd");
//       if (cudaSuccess != cudaMalloc((void **)&massl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate massd\n");  CudaTest("couldn't allocate massd");
//       if (cudaSuccess != cudaMalloc((void **)&posxl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate posxd\n");  CudaTest("couldn't allocate posxd");
//       if (cudaSuccess != cudaMalloc((void **)&posyl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate posyd\n");  CudaTest("couldn't allocate posyd");
//       if (cudaSuccess != cudaMalloc((void **)&velxl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate velxd\n");  CudaTest("couldn't allocate velxd");
//       if (cudaSuccess != cudaMalloc((void **)&velyl, sizeof(float) * (nnodes+1))) fprintf(stderr, "could not allocate velyd\n");  CudaTest("couldn't allocate velyd");
//       if (cudaSuccess != cudaMalloc((void **)&countl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate countd\n");  CudaTest("couldn't allocate countd");
//       if (cudaSuccess != cudaMalloc((void **)&startl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate startd\n");  CudaTest("couldn't allocate startd");
//       if (cudaSuccess != cudaMalloc((void **)&sortl, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate sortd\n");  CudaTest("couldn't allocate sortd");
//       if (cudaSuccess != cudaMalloc((void **)&norml, sizeof(int) * (nnodes+1))) fprintf(stderr, "could not allocate normd\n");  CudaTest("couldn't allocate normd");

//       if (cudaSuccess != cudaMalloc((void **)&maxxl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate maxxd\n");  CudaTest("couldn't allocate maxxd");
//       if (cudaSuccess != cudaMalloc((void **)&maxyl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate maxyd\n");  CudaTest("couldn't allocate maxyd");
//       if (cudaSuccess != cudaMalloc((void **)&minxl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate minxd\n");  CudaTest("couldn't allocate minxd");
//       if (cudaSuccess != cudaMalloc((void **)&minyl, sizeof(float) * blocks * FACTOR1)) fprintf(stderr, "could not allocate minyd\n");  CudaTest("couldn't allocate minyd");
//     }

//     // generate input

//     drndset(7);
//     rsc = (3 * 3.1415926535897932384626433832795) / 16;
//     vsc = sqrt(1.0 / rsc);
//     for (i = 0; i < nbodies; i++) {
//       mass[i] = 1.0 / nbodies;
//       r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
//       do {
//         x = drnd()*2.0 - 1.0;
//         y = drnd()*2.0 - 1.0;
//         sq = x*x + y*y;
//       } while (sq > 1.0);
//       scale = rsc * r / sqrt(sq);
//       posx[i] = x * scale;
//       posy[i] = y * scale;

//       do {
//         x = drnd();
//         y = drnd() * 0.1;
//       } while (y > x*x * pow(1 - x*x, 3.5));
//       v = x * sqrt(2.0 / sqrt(1 + r*r));
//       do {
//         x = drnd()*2.0 - 1.0;
//         y = drnd()*2.0 - 1.0;
//         sq = x*x + y*y;
//       } while (sq > 1.0);
//       scale = vsc * v / sqrt(sq);
//       velx[i] = x * scale;
//       vely[i] = y * scale;
//     }

//     if (cudaSuccess != cudaMemcpy(massl, mass, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of mass to device failed\n");  CudaTest("mass copy to device failed");
//     if (cudaSuccess != cudaMemcpy(posxl, posx, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posx to device failed\n");  CudaTest("posx copy to device failed");
//     if (cudaSuccess != cudaMemcpy(posyl, posy, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posy to device failed\n");  CudaTest("posy copy to device failed");
//     if (cudaSuccess != cudaMemcpy(velxl, velx, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of velx to device failed\n");  CudaTest("velx copy to device failed");
//     if (cudaSuccess != cudaMemcpy(velyl, vely, sizeof(float) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of vely to device failed\n");  CudaTest("vely copy to device failed");

//     // run timesteps (launch GPU kernels)

//     cudaEventCreate(&start);  cudaEventCreate(&stop);  
//     struct timeval starttime, endtime;
//     gettimeofday(&starttime, NULL);

//     cudaEventRecord(start, 0);
//     InitializationKernel<<<1, 1>>>(errl);
//     cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
//     timing[0] += time;
//     CudaTest("kernel 0 launch failed");

//     for (step = 0; step < timesteps; step++) {
//       cudaEventRecord(start, 0);
//       BoundingBoxKernel<<<blocks * FACTOR1, THREADS1>>>(nnodes, nbodies, startl, childl, massl, posxl, posyl, maxxl, maxyl, minxl, minyl);
//       cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
//       timing[1] += time;
//       CudaTest("kernel 1 launch failed");

//       cudaEventRecord(start, 0);
//       ClearKernel1<<<blocks * 1, 1024>>>(nnodes, nbodies, childl);
//       TreeBuildingKernel<<<blocks * FACTOR2, THREADS2>>>(nnodes, nbodies, errl, childl, posxl, posyl);
//       ClearKernel2<<<blocks * 1, 1024>>>(nnodes, startl, massl);
//       cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
//       timing[2] += time;
//       CudaTest("kernel 2 launch failed");

//       cudaEventRecord(start, 0);
//       SummarizationKernel<<<blocks * FACTOR3, THREADS3>>>(nnodes, nbodies, countl, childl, massl, posxl, posyl);
//       cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
//       timing[3] += time;
//       CudaTest("kernel 3 launch failed");

//       cudaEventRecord(start, 0);
//       SortKernel<<<blocks * FACTOR4, THREADS4>>>(nnodes, nbodies, sortl, countl, startl, childl);
//       cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
//       timing[4] += time;
//       CudaTest("kernel 4 launch failed");

//       cudaEventRecord(start, 0);
//       ForceCalculationKernel<<<blocks * FACTOR5, THREADS5>>>(nnodes, nbodies, errl, itolsq, epssq, sortl, childl, massl, posxl, posyl, velxl, velyl, norml);
//       cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
//       timing[5] += time;
//       CudaTest("kernel 5 launch failed");

//       cudaEventRecord(start, 0);
//       IntegrationKernel<<<blocks * FACTOR6, THREADS6>>>(nbodies, dtime, posxl, posyl, velxl, velyl);
//       cudaEventRecord(stop, 0);  cudaEventSynchronize(stop);  cudaEventElapsedTime(&time, start, stop);
//       timing[6] += time;
//       CudaTest("kernel 6 launch failed");
//     }
//     CudaTest("kernel launch failed");
//     cudaEventDestroy(start);  cudaEventDestroy(stop);

//     // transfer result back to CPU
//     if (cudaSuccess != cudaMemcpy(&error, errl, sizeof(int), cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of err from device failed\n");  CudaTest("err copy from device failed");
//     if (cudaSuccess != cudaMemcpy(posx, posxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posx from device failed\n");  CudaTest("posx copy from device failed");
//     if (cudaSuccess != cudaMemcpy(posy, posyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posy from device failed\n");  CudaTest("posy copy from device failed");
//     if (cudaSuccess != cudaMemcpy(velx, velxl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of velx from device failed\n");  CudaTest("velx copy from device failed");
//     if (cudaSuccess != cudaMemcpy(vely, velyl, sizeof(float) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of vely from device failed\n");  CudaTest("vely copy from device failed");

//     gettimeofday(&endtime, NULL);
//     runtime = endtime.tv_sec + endtime.tv_usec/1000000.0 - starttime.tv_sec - starttime.tv_usec/1000000.0;

//     printf("runtime: %.4lf s  (", runtime);
//     time = 0;
//     for (i = 1; i < 7; i++) {
//       printf(" %.1f ", timing[i]);
//       time += timing[i];
//     }
//     if (error == 0) {
//       printf(") = %.1f ms\n", time);
//     } else {
//       printf(") = %.1f ms FAILED %d\n", time, error);
//     }
//   }

//   // print output
//   i = 0;
// //  for (i = 0; i < nbodies; i++) {
//     printf("%.2e %.2e\n", posx[i], posy[i]);
// //  }

//   free(mass);
//   free(posx);
//   free(posy);
//   free(velx);
//   free(vely);

//   cudaFree(errl);
//   cudaFree(childl);
//   cudaFree(massl);
//   cudaFree(posxl);
//   cudaFree(posyl);
//   cudaFree(countl);
//   cudaFree(startl);
//   cudaFree(norml);

//   cudaFree(maxxl);
//   cudaFree(maxyl);
//   cudaFree(minxl);
//   cudaFree(minyl);

//   return 0;
// }
