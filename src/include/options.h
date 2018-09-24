/**
 * @brief Options header containing both options objects
  * 
 * @file options.h
 * @author David Chan
 * @date 2018-05-11
 */

#ifndef SRC_INCLUDE_OPTIONS_H_
#define SRC_INCLUDE_OPTIONS_H_

#include <random>
#include <time.h>

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

            // Random information
            int random_seed = 0;


            // Various Constructors
            Options() {}
            Options(float* return_data, float* points, int num_points, int num_dims) : 
                return_data(return_data), points(points), num_points(num_points),
                        num_dims(num_dims) {this->random_seed = time(NULL);}
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
                    {this->random_seed = time(NULL);}

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

            
    };  // End Options

    class GpuOptions {

    public:

        // GPU Options
        int device = 0;

        // Factor/thread optimization options
        int integration_kernel_threads;
        int integration_kernel_factor;
        int repulsive_kernel_threads;
        int repulsive_kernel_factor;
        int bounding_kernel_threads;
        int bounding_kernel_factor;
        int tree_kernel_threads;
        int tree_kernel_factor;
        int sort_kernel_threads;
        int sort_kernel_factor;
        int summary_kernel_threads;
        int summary_kernel_factor;
        int warp_size;
        int sm_count;

        // Constructors
        GpuOptions(int device) {
            // Setup defaults based on CUDA device
            cudaDeviceProp device_properties;
            cudaGetDeviceProperties(&device_properties, device);

            // Set the device to be used
            cudaSetDevice(device);

            // Set some base variables
            this->warp_size = device_properties.warpSize;
            if (this->warp_size != 32){
                 std::cerr << "E: Device warp size not supported." << std::endl;
                 exit(1);
            }
            this->sm_count = device_properties.multiProcessorCount;

            // Set some per-architecture structures
            // if (device_properties.major >= 6) {  // PASCAL/VOLTA

            //     this->integration_kernel_threads    = 1024;
            //     this->integration_kernel_factor     = 2;
            //     this->repulsive_kernel_threads      = 1024;
            //     this->repulsive_kernel_factor       = 2;
            //     this->bounding_kernel_threads       = 512;
            //     this->bounding_kernel_factor        = 3;
            //     this->tree_kernel_threads           = 1024;
            //     this->tree_kernel_factor            = 2;
            //     this->sort_kernel_threads           = 128;
            //     this->sort_kernel_factor            = 4;
            //     this->summary_kernel_threads        = 768;
            //     this->summary_kernel_factor         = 1;

            // } else 
            if (device_properties.major >= 5) {  // MAXWELL

                this->integration_kernel_threads    = 1024;
                this->integration_kernel_factor     = 1;
                this->repulsive_kernel_threads      = 256;
                this->repulsive_kernel_factor       = 5;
                this->bounding_kernel_threads       = 512;
                this->bounding_kernel_factor        = 3;
                this->tree_kernel_threads           = 1024;
                this->tree_kernel_factor            = 2;
                this->sort_kernel_threads           = 64;
                this->sort_kernel_factor            = 6;
                this->summary_kernel_threads        = 128;
                this->summary_kernel_factor         = 6;

            } else if (device_properties.major >= 3) {  // KEPLER

                this->integration_kernel_threads    = 1024;
                this->integration_kernel_factor     = 2;
                this->repulsive_kernel_threads      = 1024;
                this->repulsive_kernel_factor       = 2;
                this->bounding_kernel_threads       = 1024;
                this->bounding_kernel_factor        = 2;
                this->tree_kernel_threads           = 1024;
                this->tree_kernel_factor            = 2;
                this->sort_kernel_threads           = 128;
                this->sort_kernel_factor            = 4;
                this->summary_kernel_threads        = 768;
                this->summary_kernel_factor         = 1;

            } else {  // DEFAULT

                this->integration_kernel_threads    = 512;
                this->integration_kernel_factor     = 3;
                this->repulsive_kernel_threads      = 256;
                this->repulsive_kernel_factor       = 5;
                this->bounding_kernel_threads       = 512;
                this->bounding_kernel_factor        = 3;
                this->tree_kernel_threads           = 512;
                this->tree_kernel_factor            = 3;
                this->sort_kernel_threads           = 64;
                this->sort_kernel_factor            = 6;
                this->summary_kernel_threads        = 128;
                this->summary_kernel_factor         = 6;

            }
        }
    };  // End GPU Options

}

#endif  // SRC_INCLUDE_OPTIONS_H_
