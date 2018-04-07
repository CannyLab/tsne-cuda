
#ifndef PYMODULE_EXT_H
#define PYMODULE_EXT_H

    #include <sys/types.h>
    #include "common.h"
    #include "util/distance_utils.h"
    #include "naive_tsne.h"

    extern "C" {
        void pymodule_e_dist(float* points, float* dist, ssize_t *dims);
        void pymodule_naive_tsne(float *points, float *result, ssize_t *dimsm, int proj_dim, float learning_rate, float perplexity);
    }

#endif