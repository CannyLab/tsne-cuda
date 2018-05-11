/**
 * @brief Barnes-Hut T-SNE implementation O(Nlog(N))
  * 
 * @file bh_tsne.h
 * @author David Chan
 * @date 2018-04-15
 */

#ifndef BH_TSNE_H
#define BH_TSNE_H

#include "common.h"
#include "util/cuda_utils.h"
#include "util/math_utils.h"
#include "util/matrix_broadcast_utils.h"
#include "util/reduce_utils.h"
#include "util/distance_utils.h"
#include "util/random_utils.h"
#include "util/thrust_utils.h"
#include "include/util/thrust_transform_functions.h"

#include "include/kernels/apply_forces.h"
#include "include/kernels/bh_attr_forces.h"
#include "include/kernels/bh_rep_forces.h"
#include "include/kernels/bounding_box.h"
#include "include/kernels/initialization.h"
#include "include/kernels/perplexity_search.h"
#include "include/kernels/tree_builder.h"
#include "include/kernels/tree_sort.h"
#include "include/kernels/tree_summary.h"

namespace tsnecuda {

    enum TSNE_INIT {
        UNIFORM, GAUSSIAN, RESUME, VECTOR
    };

    enum RETURN_STYLE {
        ONCE, SNAPSHOT
    };
    
    class Options {

        private:
            // Dump Points Output
            bool dump_points = false;
            int dump_interval = -1;
            std::string dump_file = "";

            // Visualization
            bool use_interactive = false;
            std::string viz_server = "tcp://localhost:5556";
            int viz_timeout = 10000;

        public:
            // Point information
            /*NECESSARY*/ float* points = nullptr;
            /*NECESSARY*/ int num_points = 0;
            /*NECESSARY*/ int num_dims = 0;

            // Algorithm options
            float perplexity = 50.0f;
            float learning_rate = 200.0f;
            float early_exaggeration = 2.0f;
            float magnitude_factor = 5.0f;
            int num_neighbors = 1023;
            int iterations = 1000;
            int iterations_no_progress = 1000;
            int force_magnify_iters = 250;
            float perplexity_search_epsilon = 1e-4;

            float pre_exaggeration_momentum = 0.5;
            float post_exaggeration_momentum = 0.8;
            float theta = 0.5f;
            float epssq = 0.05 * 0.05;

            float min_gradient_norm = 0.0;

            // Initialization
            TSNE_INIT initialization = TSNE_INIT::GAUSSIAN;
            float* preinit_data = nullptr;

            // Verbosity control
            int verbosity = 20;
            int print_interval= 10;
            
            // Return methods
            RETURN_STYLE return_style = RETURN_STYLE::ONCE;
            /*NECESSARY*/ float* return_data = nullptr;
            int num_snapshots = 0; //TODO: Allow for evenly spaced snapshots

            // Editable by the tsne method
            int num_nodes = -1;
            float trained_norm = -1.0;
            bool trained = false;

            // Various Constructors
            Options() {}
            Options(float* return_data, float* points, int num_points, int num_dims) : 
                return_data(return_data), points(points), num_points(num_points),
                        num_dims(num_dims) {}
            Options(float* points, int num_points, int num_dims, 
                    float perplexity, float learning_rate, float magnitude_factor, int num_neighbors,
                    int iterations, int iterations_no_progress, int force_magnify_iters, float perplexity_search_epsilon, float pre_exaggeration_momentum, float post_exaggeration_momentum, float theta, float epssq, float min_gradient_norm,
                    TSNE_INIT initialization, float* preinit_data, 
                    bool dump_points, int dump_interval,
                    RETURN_STYLE return_style, float* return_data, int num_snapshots,
                    bool use_interactive, std::string viz_server,
                    int verbosity, int print_interval
                    ) :
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
                    initialization(initialization),
                    preinit_data(preinit_data),
                    dump_points(dump_points),
                    dump_interval(dump_interval),
                    return_style(return_style),
                    return_data(return_data),
                    num_snapshots(num_snapshots),
                    use_interactive(use_interactive),
                    viz_server(viz_server),
                    verbosity(verbosity),
                    print_interval(print_interval)
                    {}

            bool enable_dump(std::string filename, int interval = 1) {
                this->dump_points = true;
                this->dump_file = filename;
                this->dump_interval = interval;
                return true;
            }
            bool enable_viz(std::string server_address = "tcp://localhost:5556", int viz_timeout = 10000) {
                this->use_interactive = true;
                this->viz_server = server_address;
                this->viz_timeout = viz_timeout;
                return true;
            }
            bool validate() {
                if (this->points == nullptr) return false;
                if (this->num_points == 0) return false;
                if (this->num_dims == 0) return false;
                if (this->num_snapshots < 2 && this->return_style == RETURN_STYLE::SNAPSHOT) {
                    std::cout << "E: Need to record more than 1 snapshot when using snapshot capture. Use 'once' capture if you only want one return." << std::endl;
                    return false;
                }

                // Perhaps in the future this will be more exciting
                // and do much cleaner evaluation
                return true;
            }

            // Accessors for private members
            bool get_dump_points() {return this->dump_points;}
            std::string get_dump_file() {return this->dump_file;}
            int get_dump_interval() {return this->dump_interval;}
            bool get_use_interactive() {return this->use_interactive;}
            int get_viz_timeout() {return this->viz_timeout;}
            std::string get_viz_server() {return this->viz_server;}

            
    };
      
namespace bh {
void RunTsne(cublasHandle_t &dense_handle, 
                            cusparseHandle_t &sparse_handle, 
                            tsnecuda::Options &opt);
}
}


#endif
