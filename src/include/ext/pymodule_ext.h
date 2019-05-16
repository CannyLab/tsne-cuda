
#ifndef PYMODULE_EXT_H
#define PYMODULE_EXT_H

    #include <sys/types.h>
    #include "common.h"
    #include "include/options.h"
    #include "util/distance_utils.h"
    #include "bh_tsne.h"

    extern "C" {
        /**
         * @brief Exposed array for the computation of the pairwise euclidean distance
         *
         * @param points The points to compute the distance on, Column-Major NxNDIM array
         * @param dist The output distance array
         * @param dims The dimensions of the points matrix
         */
        // void pymodule_e_dist(float* points, float* dist, ssize_t *dims);

        /**
         * @brief Exposed method for computing the pij array in the external python module
         *
         * @param points The points to compute, Column-Major NxNDIM array
         * @param sigmas The relevant sigmas
         * @param result The result array
         * @param dimsm The dimensions of the points array
         */
        // void pymodule_compute_pij(float *points, float* sigmas, float *result, ssize_t *dimsm);

	void pymodule_bh_tsne(float *result,
                      float* points,
                      ssize_t *dims,
                      float perplexity,
                      float learning_rate,
                      float magnitude_factor,
                      int num_neighbors,
                      int iterations,
                      int iterations_no_progress,
                      int force_magnify_iters,
                      float perplexity_search_epsilon,
                      float pre_exaggeration_momentum,
                      float post_exaggeration_momentum,
                      float theta,
                      float epssq,
                      float min_gradient_norm,
                      int initialization_type,
                      float* preinit_data,
                      bool dump_points,
                      char* dump_file,
                      int dump_interval,
                      bool use_interactive,
                      char* viz_server,
                      int viz_timeout,
                      int verbosity,
                      int print_interval,
                      int gpu_device,
                      int return_style,
                      int num_snapshots
                 );

    }

#endif
