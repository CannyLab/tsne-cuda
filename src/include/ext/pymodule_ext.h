
#ifndef PYMODULE_EXT_H
#define PYMODULE_EXT_H

    #include <sys/types.h>
    #include "common.h"
    #include "util/distance_utils.h"
    #include "naive_tsne.h"

    extern "C" {
        /**
         * @brief Exposed array for the computation of the pairwise euclidean distance
         * 
         * @param points The points to compute the distance on, Column-Major NxNDIM array
         * @param dist The output distance array
         * @param dims The dimensions of the points matrix
         */
        void pymodule_e_dist(float* points, float* dist, ssize_t *dims);
        
        /**
         * @brief Exposed method for computing the pij array in the external python module
         * 
         * @param points The points to compute, Column-Major NxNDIM array
         * @param sigmas The relevant sigmas
         * @param result The result array
         * @param dimsm The dimensions of the points array
         */
        void pymodule_compute_pij(float *points, float* sigmas, float *result, ssize_t *dimsm);

        /**
         * @brief Exposed method for computing naive T-SNE in the external python module
         * 
         * @param points The points to compute, Column-Major NxNDIM array
         * @param result The result array
         * @param dimsm The dimensions of the points array
         * @param proj_dim The number of dimensions to project to
         * @param learning_rate The learning rate of the gradient ascent
         * @param perplexity The target perplexity
         */
        void pymodule_naive_tsne(float *points, float *result, ssize_t *dimsm, int proj_dim, float learning_rate, float perplexity);
    }

#endif