/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @brief Options header containing both options objects
  *
 * @file options.h
 * @author David Chan
 * @date 2018-05-11
 */

#ifndef SRC_INCLUDE_OPTIONS_H_
#define SRC_INCLUDE_OPTIONS_H_

#include <sycl/sycl.hpp>
#include <random>
#include <time.h>
#include <iostream>

namespace faiss {
/// The metric space for vector comparison for Faiss indices and algorithms.
///
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.
enum MetricType {
    METRIC_INNER_PRODUCT = 0, ///< maximum inner product search
    METRIC_L2 = 1,            ///< squared L2 search
    METRIC_L1,                ///< L1 (aka cityblock)
    METRIC_Linf,              ///< infinity distance
    METRIC_Lp,                ///< L_p distance, p is given by a faiss::Index
                              /// metric_arg

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Canberra = 20,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,
};
}

namespace tsnecuda
{
    enum TSNE_INIT
    {
        UNIFORM,
        GAUSSIAN,
        RESUME,
        VECTOR
    };

    enum RETURN_STYLE
    {
        ONCE,
        SNAPSHOT
    };

    enum DISTANCE_METRIC
    {
        INNER_PRODUCT,
        L2,
        L1,
        LINF,
        CANBERRA,
        BRAYCURTIS,
        JENSENSHANNON,
    };

    class Options
    {

    private:
        // Dump Points Output
        bool dump_points        = true;
        int dump_interval       = 999;
        std::string dump_file   = "tsne_mnist_output.txt";

        // Visualization
        bool use_interactive    = false;
        std::string viz_server  = "tcp://localhost:5556";
        int viz_timeout         = 10000;

    public:
        // Point information
        float *points   = nullptr;  /*NECESSARY*/
        int num_points  = 0;        /*NECESSARY*/
        int num_dims    = 0;        /*NECESSARY*/

        // Algorithm options
        float perplexity                    = 50.0f;
        float learning_rate                 = 200.0f;
        float early_exaggeration            = 12.0f;
        float magnitude_factor              = 5.0f;
        int num_neighbors                   = 1023;
        int iterations                      = 1000;
        int iterations_no_progress          = 1000;
        int force_magnify_iters             = 250;
        float perplexity_search_epsilon     = 1e-4f;
        float pre_exaggeration_momentum     = 0.5f;
        float post_exaggeration_momentum    = 0.8f;
        float theta                         = 0.5f;
        float epssq                         = 0.05f * 0.05f;
        float min_gradient_norm             = 0.0f;

        // Distances
        faiss::MetricType distance_metric = faiss::METRIC_INNER_PRODUCT;

        // Initialization
        TSNE_INIT initialization    = TSNE_INIT::GAUSSIAN;
        float *preinit_data         = nullptr;

        // Verbosity control
        int verbosity       = 20;
        int print_interval  = 10;

        // Return methods
        RETURN_STYLE return_style   = RETURN_STYLE::ONCE;
        float *return_data          = nullptr;  /*NECESSARY*/
        int num_snapshots           = 0;        //TODO: Allow for evenly spaced snapshots

        // Editable by the tsne method
        float trained_norm  = -1.0f;
        bool trained        = false;

        // Random information
        int64_t random_seed = 0;

        // Various Constructors
        Options() {}

        Options(
            float* points,      // image data
            int num_points,     // number of images
            int num_dims,       // number of data points (pixels) per image
            float* return_data) :
                points(points),
                num_points(num_points),
                num_dims(num_dims),
                return_data(return_data)
        { this->random_seed = time(NULL); }

        Options(
            bool dump_points,
            int dump_interval,
            bool use_interactive,
            std::string viz_server,
            float* points,
            int num_points,
            int num_dims,
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
            faiss::MetricType distance_metric,
            TSNE_INIT initialization,
            float* preinit_data,
            int verbosity,
            int print_interval,
            RETURN_STYLE return_style,
            float *return_data,
            int num_snapshots) :
                dump_points(dump_points),
                dump_interval(dump_interval),
                use_interactive(use_interactive),
                viz_server(viz_server),
                points(points),
                num_points(num_points),
                num_dims(num_dims),
                perplexity(perplexity),
                learning_rate(learning_rate),
                magnitude_factor(magnitude_factor),
                num_neighbors(num_neighbors),
                iterations(iterations),
                iterations_no_progress(iterations_no_progress),
                force_magnify_iters(force_magnify_iters),
                perplexity_search_epsilon(perplexity_search_epsilon),
                pre_exaggeration_momentum(pre_exaggeration_momentum),
                post_exaggeration_momentum(post_exaggeration_momentum),
                theta(theta),
                epssq(epssq),
                min_gradient_norm(min_gradient_norm),
                distance_metric(distance_metric),
                initialization(initialization),
                preinit_data(preinit_data),
                verbosity(verbosity),
                print_interval(print_interval),
                return_style(return_style),
                return_data(return_data),
                num_snapshots(num_snapshots)
        { this->random_seed = time(NULL); }

        bool enable_dump(std::string filename, int interval = 1)
        {
            this->dump_points = true;
            this->dump_file = std::move(filename);
            this->dump_interval = interval;
            return true;
        }

        bool enable_viz(std::string server_address = "tcp://localhost:5556", int viz_timeout = 10000)
        {
            this->use_interactive = true;
            this->viz_server = std::move(server_address);
            this->viz_timeout = viz_timeout;
            return true;
        }

        bool validate()
        {
            if (this->num_points == 0)
                return false;
            if (this->num_dims == 0)
                return false;
            if (this->num_snapshots < 2 && this->return_style == RETURN_STYLE::SNAPSHOT)
            {
                std::cout << "E: Need to record more than 1 snapshot when using snapshot capture. "
                          << "Use 'once' capture if you only want one return." << std::endl;
                return false;
            }

            // Perhaps in the future this will be more exciting
            // and do much cleaner evaluation
            return true;
        }

        // Accessors for private members
        bool        get_dump_points()       { return this->dump_points;     }
        std::string get_dump_file()         { return this->dump_file;       }
        int         get_dump_interval()     { return this->dump_interval;   }
        bool        get_use_interactive()   { return this->use_interactive; }
        int         get_viz_timeout()       { return this->viz_timeout;     }
        std::string get_viz_server()        { return this->viz_server;      }

    }; // End Options

} // namespace tsnecuda

#endif // SRC_INCLUDE_OPTIONS_H_
