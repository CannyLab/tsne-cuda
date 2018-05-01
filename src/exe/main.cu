
// This file exposes a main file which does most of the testing with command line 
// args, so we don't have to re-build to change options.

// Detailed includes
#include "common.h"
#include "util/data_utils.h"
#include "util/cuda_utils.h"
#include "util/random_utils.h"
#include "util/distance_utils.h"
#include "naive_tsne.h"
#include "naive_tsne_cpu.h"
#include "bh_tsne_ref.h"
#include "bh_tsne.h"
#include <time.h>
#include <string>

// Option parser
#include <cxxopts.hpp>

#define STRINGIFY(X) #X

#define FOPT(x) result[STRINGIFY(x)].as<float>()
#define SOPT(x) result[STRINGIFY(x)].as<std::string>()
#define IOPT(x) result[STRINGIFY(x)].as<int>()
#define BOPT(x) result[STRINGIFY(x)].as<bool>()



int main(int argc, char** argv) {

    // Setup command line options
    cxxopts::Options options("TSNE-CUDA","Perform T-SNE in an optimized manner.");
    options.add_options()
        ("l,learning-rate", "Learning Rate", cxxopts::value<float>()->default_value("200"))
        ("p,perplexity", "Perplexity", cxxopts::value<float>()->default_value("35.0"))
        ("e,early-ex", "Early Exaggeration Factor", cxxopts::value<float>()->default_value("12.0"))
        ("s,data", "Which program to run on <cifar10,cifar100,mnist,sim>", cxxopts::value<std::string>()->default_value("sim"))
        ("k,num-points", "How many simulated points to use", cxxopts::value<int>()->default_value("5000"))
        ("n,num-steps", "How many steps to take", cxxopts::value<int>()->default_value("1000"))
        ("i,viz", "Use interactive visualization", cxxopts::value<bool>()->default_value("false"))
        ("d,dump", "Dump the output points", cxxopts::value<bool>()->default_value("false"))
        ("m,magnitude-factor", "Magnitude factor for KNN", cxxopts::value<float>()->default_value("5.0"))
        ("t,init", "What kind of initialization to use <unif,gauss>", cxxopts::value<std::string>()->default_value("unif"))
        ("f,fname", "File name for loaded data...", cxxopts::value<std::string>()->default_value("../train-images.idx3-ubyte"));
    
    // Parse command line options
    auto result = options.parse(argc, argv);

    // Common initialization
    srand (time(NULL));

    // --- Matrices allocation and initialization
    cublasHandle_t dense_handle;
    cublasSafeCall(cublasCreate(&dense_handle));
    cusparseHandle_t sparse_handle;
    cusparseSafeCall(cusparseCreate(&sparse_handle));

    int init_type = 0;
    if (SOPT(init).compare("unif") == 0) {
        init_type = 0;
    } else {
        init_type = 1;
    }


    if (SOPT(data).compare("mnist") == 0) {

        // Load the data
        int num_images, num_columns, num_rows;
        float* data = Data::load_mnist(SOPT(fname), num_images, num_columns, num_rows);

        // Do the T-SNE
        printf("Starting TSNE calculation with %u points.\n", num_images);
        BHTSNE::tsne(dense_handle, sparse_handle, data, num_images, num_columns*num_rows,
                        2, FOPT(perplexity), FOPT(learning-rate), FOPT(early-ex), IOPT(num-steps),IOPT(num-steps), 0.0,
                        BOPT(dump), BOPT(viz), FOPT(magnitude-factor), init_type  );

        // Clean up the data
        delete[] data;

    } else if (SOPT(data).compare("cifar10") == 0) {

        // Load the data
        int num_images = 50000;
        int num_columns = 32;
        int num_rows = 32;
        int num_channels = 3;
        float * data = Data::load_cifar10(SOPT(fname));

        // Do the T-SNE
        printf("Starting TSNE calculation with %u points.\n", num_images);
        BHTSNE::tsne(dense_handle, sparse_handle, data, num_images, num_channels*num_columns*num_rows,
                        2, FOPT(perplexity), FOPT(learning-rate), FOPT(early-ex), IOPT(num-steps),IOPT(num-steps), 0.0,
                        BOPT(dump), BOPT(viz), FOPT(magnitude-factor), init_type );

        // Clean up the data
        delete[] data;


    } else if (SOPT(data).compare("cifar100") == 0) {

        // Load the data
        int num_images = 50000;
        int num_columns = 32;
        int num_rows = 32;
        int num_channels = 3;
        float * data = Data::load_cifar100(SOPT(fname));

        // DO the T-SNE
        printf("Starting TSNE calculation with %u points.\n", num_images);
        BHTSNE::tsne(dense_handle, sparse_handle, data, num_images, num_channels*num_columns*num_rows,
                        2, FOPT(perplexity), FOPT(learning-rate), FOPT(early-ex), IOPT(num-steps),IOPT(num-steps), 0.0,
                        BOPT(dump), BOPT(viz), FOPT(magnitude-factor), init_type );
        
        // Clean up the data
        delete[] data;

    } else if (SOPT(data).compare("sim") == 0) {

        // Generate some random points in 2 clusters
        std::default_random_engine generator;
        std::normal_distribution<double> distribution1(-10.0, 1.0);
        std::normal_distribution<double> distribution2(10.0, 1.0);

        thrust::host_vector<float> h_X(50 * IOPT(num-points));
        for (int i = 0; i < 50 *  IOPT(num-points); i ++) {
            if (i < ((IOPT(num-points) / 2) * 50)) {
                h_X[i] = distribution1(generator);
            } else {
                h_X[i] = distribution2(generator);
            }
        }

        // Do the T-SNE
        printf("Starting TSNE calculation with %u points.\n",  IOPT(num-points));
        BHTSNE::tsne(dense_handle, sparse_handle, thrust::raw_pointer_cast(h_X.data()),  IOPT(num-points), 50,
                        2, FOPT(perplexity), FOPT(learning-rate), FOPT(early-ex), IOPT(num-steps),IOPT(num-steps), 0.0,
                        BOPT(dump), BOPT(viz), FOPT(magnitude-factor), init_type );


    } else {
        std::cout << "Dataset not recognized..." << std::endl;
    }

    return 0;
}