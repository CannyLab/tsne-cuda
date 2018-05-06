// TODO: add copyright

/*
    Apply forces to the points with momentum, exaggeration, etc.
*/

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
                        float exaggeration,
                        volatile float * __restrict pts, // (nnodes + 1) x 2
                        volatile float * __restrict attr_forces, // (N x 2)
                        volatile float * __restrict rep_forces, // (nnodes + 1) x 2
                        volatile float * __restrict gains,
                        volatile float * __restrict old_forces) // (N x 2)
{
  register int i, inc;
  register float dx, dy, ux, uy, gx, gy;

  // iterate over all bodies assigned to thread
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += inc) {
        ux = old_forces[i];
        uy = old_forces[N + i];
        gx = gains[i];
        gy = gains[N + i];
        dx = exaggeration*attr_forces[i] - (rep_forces[i] / norm);
        dy = exaggeration*attr_forces[i + N] - (rep_forces[nnodes + 1 + i] / norm);

        gx = (signbit(dx) != signbit(ux)) ? gx + 0.2 : gx * 0.8;
        gy = (signbit(dy) != signbit(uy)) ? gy + 0.2 : gy * 0.8;
        gx = (gx < 0.01) ? 0.01 : gx;
        gy = (gy < 0.01) ? 0.01 : gy;

        ux = momentum * ux - eta * gx * dx;
        uy = momentum * uy - eta * gy * dy;

        pts[i] += ux;
        pts[i + nnodes + 1] += uy;

        old_forces[i] = ux;
        old_forces[N + i] = uy;
        gains[i] = gx;
        gains[N + i] = gy;
   }
}
