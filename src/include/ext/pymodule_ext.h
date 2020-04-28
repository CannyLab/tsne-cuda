#ifndef PYMODULE_EXT_H
#define PYMODULE_EXT_H

#include <sys/types.h>
#include "common.h"
#include "options.h"
#include "util/distance_utils.h"
#include "fit_tsne.h"

extern "C"
{
    void pymodule_tsne(float *result,
                       float *points,
                       ssize_t *dims,
                       float perplexity,
                       float learning_rate,
                       float early_exaggeration,
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
                       float *preinit_data,
                       bool dump_points,
                       char *dump_file,
                       int dump_interval,
                       bool use_interactive,
                       char *viz_server,
                       int viz_timeout,
                       int verbosity,
                       int print_interval,
                       int gpu_device,
                       int return_style,
                       int num_snapshots,
                       int distance_metric);
}

#endif
