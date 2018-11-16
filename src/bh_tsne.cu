/*
    Compute t-SNE via Barnes-Hut for NlogN time.
*/

#include "bh_tsne.h"
#include "FIt-SNE/src/nbodyfft.h"

float squared_cauchy_2d(float x1, float x2, float y1, float y2) {
    return pow(1.0 + pow(x1 - y1, 2) + pow(x2 - y2, 2), -2);
}

void tsnecuda::bh::RunTsne(tsnecuda::Options &opt,
                            tsnecuda::GpuOptions &gpu_opt)
{
    // Check the validity of the options file
    if (!opt.validate()) {
        std::cout << "E: Invalid options file. Terminating." << std::endl;
        return;
    }

    // Construct the handles
    cublasHandle_t dense_handle;
    CublasSafeCall(cublasCreate(&dense_handle));
    cusparseHandle_t sparse_handle;
    CusparseSafeCall(cusparseCreate(&sparse_handle));

    // Set CUDA device properties
    const int num_blocks = gpu_opt.sm_count;
    
    // Construct sparse matrix descriptor
    cusparseMatDescr_t sparse_matrix_descriptor;
    cusparseCreateMatDescr(&sparse_matrix_descriptor);
    cusparseSetMatType(sparse_matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(sparse_matrix_descriptor,CUSPARSE_INDEX_BASE_ZERO);
    
    // Setup some return information if we're working on snapshots
    int snap_interval;
    int snap_num = 0;
    if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT) {
      snap_interval = opt.iterations / (opt.num_snapshots-1);
    }

    // Get constants from options
    const int num_points = opt.num_points;
    const int num_neighbors = (opt.num_neighbors < num_points) ? opt.num_neighbors : num_points;
    const float *high_dim_points = opt.points;
    const int high_dim = opt.num_dims;
    const float perplexity = opt.perplexity;
    const float perplexity_search_epsilon = opt.perplexity_search_epsilon;
    const float eta = opt.learning_rate;
    float momentum = opt.pre_exaggeration_momentum;
    float attr_exaggeration = opt.early_exaggeration;
    float normalization;

    // Theta governs tolerance for Barnes-Hut recursion
    const float theta = opt.theta;
    const float epsilon_squared = opt.epssq;

    // Figure out number of nodes needed for BH tree
    int nnodes = num_points * 2;
    if (nnodes < 1024 * num_blocks) nnodes = 1024 * num_blocks;
    while ((nnodes & (gpu_opt.warp_size - 1)) != 0)
        nnodes++;
    nnodes--;
    const int num_nodes = nnodes;
    opt.num_nodes = num_nodes;
    
    // Allocate host memory
    float *knn_squared_distances = new float[num_points * num_neighbors];
    memset(knn_squared_distances, 0, num_points * num_neighbors * sizeof(float));
    long *knn_indices = new long[num_points * num_neighbors];

    // Initialize global variables
    thrust::device_vector<int> err_device(1);
    tsnecuda::bh::Initialize(gpu_opt, err_device);

    // Compute approximate K Nearest Neighbors and squared distances
    tsnecuda::util::KNearestNeighbors(gpu_opt, knn_indices, knn_squared_distances, high_dim_points, high_dim, num_points, num_neighbors);
    thrust::device_vector<long> knn_indices_long_device(knn_indices, knn_indices + num_points * num_neighbors);
    thrust::device_vector<int> knn_indices_device(num_points * num_neighbors);
    tsnecuda::util::PostprocessNeighborIndices(gpu_opt, knn_indices_device, knn_indices_long_device, 
                                                        num_points, num_neighbors);
    
    // Max-norm the distances to avoid exponentiating by large numbers
    thrust::device_vector<float> knn_squared_distances_device(knn_squared_distances, 
                                            knn_squared_distances + (num_points * num_neighbors));
    tsnecuda::util::MaxNormalizeDeviceVector(knn_squared_distances_device);

    // Search Perplexity
    thrust::device_vector<float> pij_non_symmetric_device(num_points * num_neighbors);
    tsnecuda::bh::SearchPerplexity(gpu_opt, dense_handle, pij_non_symmetric_device, knn_squared_distances_device, 
                                    perplexity, perplexity_search_epsilon, num_points, num_neighbors);

    // Clean up memory
    knn_squared_distances_device.clear();
    knn_squared_distances_device.shrink_to_fit();
    knn_indices_long_device.clear();
    knn_indices_long_device.shrink_to_fit();
    delete[] knn_squared_distances;
    delete[] knn_indices;

    // Symmetrize the pij matrix
    thrust::device_vector<float> sparse_pij_device;
    thrust::device_vector<int> pij_row_ptr_device;
    thrust::device_vector<int> pij_col_ind_device;
    tsnecuda::util::SymmetrizeMatrix(sparse_handle, sparse_pij_device, pij_row_ptr_device,
                                        pij_col_ind_device, pij_non_symmetric_device, knn_indices_device,
                                        opt.magnitude_factor, num_points, num_neighbors);

    const int num_nonzero = sparse_pij_device.size();

    // Clean up memory
    knn_indices_device.clear();
    knn_indices_device.shrink_to_fit();
    pij_non_symmetric_device.clear();
    pij_non_symmetric_device.shrink_to_fit();

    // Declare memory
    thrust::device_vector<float> pij_x_qij_device(sparse_pij_device.size());
    thrust::device_vector<float> repulsive_forces_device((num_nodes + 1) * 2, 0);
    thrust::device_vector<float> attractive_forces_device(opt.num_points * 2, 0);
    thrust::device_vector<float> gains_device(opt.num_points * 2, 1);
    thrust::device_vector<float> old_forces_device(opt.num_points * 2, 0); // for momentum
    thrust::device_vector<int> cell_starts_device(num_nodes + 1);
    thrust::device_vector<int> children_device((num_nodes + 1) * 4);
    thrust::device_vector<float> cell_mass_device(num_nodes + 1, 1.0); // TODO: probably don't need massl
    thrust::device_vector<int> cell_counts_device(num_nodes + 1);
    thrust::device_vector<int> cell_sorted_device(num_nodes + 1);
    thrust::device_vector<float> normalization_vec_device(num_nodes + 1);
    thrust::device_vector<float> x_max_device(num_blocks * gpu_opt.bounding_kernel_factor);
    thrust::device_vector<float> y_max_device(num_blocks * gpu_opt.bounding_kernel_factor);
    thrust::device_vector<float> x_min_device(num_blocks * gpu_opt.bounding_kernel_factor);
    thrust::device_vector<float> y_min_device(num_blocks * gpu_opt.bounding_kernel_factor);
    thrust::device_vector<float> ones_device(opt.num_points * 2, 1); // This is for reduce summing, etc.
    thrust::device_vector<int> coo_indices_device(sparse_pij_device.size()*2);

    tsnecuda::util::Csr2Coo(gpu_opt, coo_indices_device, pij_row_ptr_device,
                            pij_col_ind_device, num_points, num_nonzero);

    // Initialize Low-Dim Points
    thrust::device_vector<float> points_device((num_nodes + 1) * 2);
    thrust::device_vector<float> random_vector_device(points_device.size());

    std::default_random_engine generator(opt.random_seed);
    std::normal_distribution<float> distribution1(0.0, 1.0);
    thrust::host_vector<float> h_points_device((num_nodes+ 1) * 2);

    // Initialize random noise vector
    for (int i = 0; i < (num_nodes+1)*2; i++) h_points_device[i] = 0.001 * distribution1(generator);
    thrust::copy(h_points_device.begin(), h_points_device.end(), random_vector_device.begin());

    // TODO: this will only work with gaussian init
    if (opt.initialization == tsnecuda::TSNE_INIT::UNIFORM) { // Random uniform initialization
        points_device = tsnecuda::util::RandomDeviceVectorInRange(generator, (num_nodes+1)*2, -5, 5);
    } else if (opt.initialization == tsnecuda::TSNE_INIT::GAUSSIAN) { // Random gaussian initialization
        // Generate some Gaussian noise for the points
        for (int i = 0; i < (num_nodes+ 1) * 2; i++) h_points_device[i] = 0.0001 * distribution1(generator);
        thrust::copy(h_points_device.begin(), h_points_device.end(), points_device.begin());
    } else if (opt.initialization == tsnecuda::TSNE_INIT::RESUME) { // Preinit from vector
        // Load from vector
        if(opt.preinit_data != nullptr) {
          thrust::copy(opt.preinit_data, opt.preinit_data+(num_nodes+1)*2, points_device.begin());
        } else {
          std::cerr << "E: Invalid initialization. Initialization points are null." << std::endl;
          exit(1);
        }
    } else if (opt.initialization == tsnecuda::TSNE_INIT::VECTOR) { // Preinit from vector points only
        // Copy the pre-init data
        if(opt.preinit_data != nullptr) {
          thrust::copy(opt.preinit_data, opt.preinit_data+opt.num_points, points_device.begin());
          thrust::copy(opt.preinit_data+opt.num_points+1, opt.preinit_data+opt.num_points*2 , points_device.begin()+(num_nodes+1));
        } else {
          std::cerr << "E: Invalid initialization. Initialization points are null." << std::endl;
          exit(1);
        }
    } else { // Invalid initialization
        std::cerr << "E: Invalid initialization type specified." << std::endl;
        exit(1);
    }
    
    // Dump file
    float *host_ys = nullptr;
    std::ofstream dump_file;
    if (opt.get_dump_points()) {
        dump_file.open(opt.get_dump_file());
        host_ys = new float[(num_nodes + 1) * 2];
        dump_file << num_points << " " << 2 << std::endl;
    }

    #ifndef NO_ZMQ 
        bool send_zmq = opt.get_use_interactive();
        zmq::context_t context(1);
        zmq::socket_t publisher(context, ZMQ_REQ);
        if (opt.get_use_interactive()) {

        // Try to connect to the socket
            if (opt.verbosity >= 1)
                std::cout << "Initializing Connection...." << std::endl;
            publisher.setsockopt(ZMQ_RCVTIMEO, opt.get_viz_timeout());
            publisher.setsockopt(ZMQ_SNDTIMEO, opt.get_viz_timeout());
            if (opt.verbosity >= 1)
                std::cout << "Waiting for connection to visualization for 10 secs...." << std::endl;
            publisher.connect(opt.get_viz_server());

            // Send the number of points we should be expecting to the server
            std::string message = std::to_string(opt.num_points);
            send_zmq = publisher.send(message.c_str(), message.length());

            // Wait for server reply
            zmq::message_t request;
            send_zmq = publisher.recv (&request);
            
            // If there's a time-out, don't bother.
            if (send_zmq) {
                if (opt.verbosity >= 1)
                    std::cout << "Visualization connected!" << std::endl;
            } else {
                std::cout << "No Visualization Terminal, continuing..." << std::endl;
                send_zmq = false;
            }
        }
    #else
        if (opt.get_use_interactive()) 
            std::cout << "This version is not built with ZMQ for interative viz. Rebuild with WITH_ZMQ=TRUE for viz." << std::endl;
    #endif

    // Support for infinite iteration
    for (size_t step = 0; step != opt.iterations; step++) {

        // Setup learning rate schedule
        if (step == opt.force_magnify_iters) {
            momentum = opt.post_exaggeration_momentum;
            attr_exaggeration = 1.0f;
        }

        // Compute Bounding Box
        tsnecuda::bh::ComputeBoundingBox(gpu_opt,
                                         cell_starts_device,
                                         children_device,
                                         cell_mass_device,
                                         points_device,
                                         x_max_device,
                                         y_max_device,
                                         x_min_device,
                                         y_min_device,
                                         num_nodes, 
                                         num_points, 
                                         num_blocks);

        // Tree Builder
        tsnecuda::bh::BuildTree(gpu_opt,
                                err_device,
                                children_device,
                                cell_starts_device,
                                cell_mass_device,
                                points_device,
                                num_nodes,
                                num_points,
                                num_blocks);

        // Tree Summarization
        tsnecuda::bh::SummarizeTree(gpu_opt,
                                    cell_counts_device,
                                    children_device,
                                    cell_mass_device,
                                    points_device,
                                    num_nodes,
                                    num_points,
                                    num_blocks);

        // Sort By Morton Code
        tsnecuda::bh::SortCells(gpu_opt,
                                cell_sorted_device,
                                cell_starts_device,
                                children_device,
                                cell_counts_device,
                                num_nodes,
                                num_points,
                                num_blocks);



        // Copy data back from GPU to CPU
        thrust::copy(points_device.begin(), points_device.end(), h_points_device.begin());

        // Hyperparameters
        int n_interpolation_points = 3;
        float intervals_per_integer = 1;
        int min_num_intervals = 50;
        
        // Compute repulsive forces

         // Zero out the gradient
        auto N = num_points;
        auto D = 2;

        // For convenience, split the x and y coordinate values
        float* xs = new float[N];
        float* ys = new float[N];

        float min_coord = INFINITY;
        float max_coord = -INFINITY;
        // Find the min/max values of the x and y coordinates
        for (unsigned long i = 0; i < N; i++) {
            xs[i] = h_points_device[i];
            ys[i] = h_points_device[i + num_nodes + 1];
            if (xs[i] > max_coord) max_coord = xs[i];
            else if (xs[i] < min_coord) min_coord = xs[i];
            if (ys[i] > max_coord) max_coord = ys[i];
            else if (ys[i] < min_coord) min_coord = ys[i];
        }

        // The number of "charges" or s+2 sums i.e. number of kernel sums
        int n_terms = 4;
        auto *chargesQij = new float[N * n_terms];
        auto *potentialsQij = new float[N * n_terms]();
        // thrust::device_vector<float> potentialsQij(N * n_terms);

        // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
        for (unsigned long j = 0; j < N; j++) {
            chargesQij[j * n_terms + 0] = 1;
            chargesQij[j * n_terms + 1] = xs[j];
            chargesQij[j * n_terms + 2] = ys[j];
            chargesQij[j * n_terms + 3] = xs[j] * xs[j] + ys[j] * ys[j];
        }

        // Compute the number of boxes in a single dimension and the total number of boxes in 2d
        auto n_boxes_per_dim = static_cast<int>(fmax(min_num_intervals, (max_coord - min_coord) / intervals_per_integer));


        // FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
        // 11^e 13^f, where e+f is either 0 or 1, and the other exponents are
        // arbitrary
        int allowed_n_boxes_per_dim[20] = {25,36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140,150, 175, 200};
        if ( n_boxes_per_dim < allowed_n_boxes_per_dim[19] ) {
            //Round up to nearest grid point
            int chosen_i;
            for (chosen_i =0; allowed_n_boxes_per_dim[chosen_i]< n_boxes_per_dim; chosen_i++);
            n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
        }

        int n_boxes = n_boxes_per_dim * n_boxes_per_dim;

        auto *box_lower_bounds = new float[2 * n_boxes];
        auto *box_upper_bounds = new float[2 * n_boxes];
        auto *y_tilde_spacings = new float[n_interpolation_points];
        int n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;
        auto *x_tilde = new float[n_interpolation_points_1d]();
        auto *y_tilde = new float[n_interpolation_points_1d]();
        auto *fft_kernel_tilde = new complex<float>[2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d];

        precompute_2d(max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim, n_interpolation_points,
                    &squared_cauchy_2d,
                    box_lower_bounds, box_upper_bounds, y_tilde_spacings, x_tilde, y_tilde, fft_kernel_tilde);
        n_body_fft_2d(N, n_terms, xs, ys, chargesQij, n_boxes_per_dim, n_interpolation_points, box_lower_bounds,
                    box_upper_bounds, y_tilde_spacings, fft_kernel_tilde, potentialsQij,12);

        // Compute the normalization constant Z or sum of q_{ij}. This expression is different from the one in the original
        // paper, but equivalent. This is done so we need only use a single kernel (K_2 in the paper) instead of two
        // different ones. We subtract N at the end because the following sums over all i, j, whereas Z contains i \neq j
        float sum_Q = 0;
        for (unsigned long i = 0; i < N; i++) {
            float phi1 = potentialsQij[i * n_terms + 0];
            float phi2 = potentialsQij[i * n_terms + 1];
            float phi3 = potentialsQij[i * n_terms + 2];
            float phi4 = potentialsQij[i * n_terms + 3];

            sum_Q += (1 + xs[i] * xs[i] + ys[i] * ys[i]) * phi1 - 2 * (xs[i] * phi2 + ys[i] * phi3) + phi4;
        }
        sum_Q -= N;

        normalization = sum_Q;

        // Make the negative term, or F_rep in the equation 3 of the paper
        float *neg_f = new float[N * 2];
        for (unsigned int i = 0; i < N; i++) {
            h_points_device[i] = (xs[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 1]);
            h_points_device[i + num_nodes + 1] = (ys[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 2]);
        }

        thrust::copy(h_points_device.begin(), h_points_device.end(), repulsive_forces_device.begin());

        // Copy data back to the GPU

        delete[] neg_f;
        delete[] potentialsQij;
        delete[] chargesQij;
        delete[] xs;
        delete[] ys;
        delete[] box_lower_bounds;
        delete[] box_upper_bounds;
        delete[] y_tilde_spacings;
        delete[] y_tilde;
        delete[] x_tilde;
        delete[] fft_kernel_tilde;
        
        



        // Calculate Repulsive Forces
        // tsnecuda::bh::ComputeRepulsiveForces(gpu_opt,
        //                                      err_device,
        //                                      repulsive_forces_device,
        //                                      normalization_vec_device,
        //                                      cell_sorted_device,
        //                                      children_device,
        //                                      cell_mass_device,
        //                                      points_device,
        //                                      theta, 
        //                                      epsilon_squared,
        //                                      num_nodes,
        //                                      num_points,
        //                                      num_blocks);
        // Compute normalization
        // normalization = thrust::reduce(normalization_vec_device.begin(),
        //                                 normalization_vec_device.end(),
        //                                 0.0f,
        //                                 thrust::plus<float>());

        // Calculate Attractive Forces
        tsnecuda::bh::ComputeAttractiveForces(gpu_opt,
                                              sparse_handle,
                                              sparse_matrix_descriptor,
                                              attractive_forces_device,
                                              pij_x_qij_device,
                                              sparse_pij_device,
                                              pij_row_ptr_device,
                                              pij_col_ind_device,
                                              coo_indices_device,
                                              points_device,
                                              ones_device,
                                              num_nodes,
                                              num_points,
                                              num_nonzero);

        // Apply Forces
        tsnecuda::bh::ApplyForces(gpu_opt,
                                  points_device,
                                  attractive_forces_device,
                                  repulsive_forces_device,
                                  gains_device,
                                  old_forces_device,
                                  eta,
                                  normalization,
                                  momentum,
                                  attr_exaggeration,
                                  num_nodes,
                                  num_points,
                                  num_blocks);
        // Add a bit of random motion to prevent points from being on top of each other
        thrust::transform(points_device.begin(), points_device.end(), random_vector_device.begin(),
                            points_device.begin(), thrust::plus<float>());

        // Compute the gradient norm
        tsnecuda::util::SquareDeviceVector(attractive_forces_device, old_forces_device);
        thrust::transform(attractive_forces_device.begin(), attractive_forces_device.begin()+num_points, 
                          attractive_forces_device.begin()+num_points, attractive_forces_device.begin(), 
                          thrust::plus<float>());
        tsnecuda::util::SqrtDeviceVector(attractive_forces_device, attractive_forces_device);
        float grad_norm = thrust::reduce(attractive_forces_device.begin(), attractive_forces_device.begin()+num_points, 
                                         0.0f, thrust::plus<float>()) / num_points;
        thrust::fill(attractive_forces_device.begin(), attractive_forces_device.end(), 0.0f);

        if (grad_norm < opt.min_gradient_norm) {
            if (opt.verbosity >= 1) std::cout << "Reached minimum gradient norm: " << grad_norm << std::endl;
            break;
        }

        if (opt.verbosity >= 1 && step % opt.print_interval == 0) {
            std::cout << "[Step " << step << "] Avg. Gradient Norm: " << grad_norm << std::endl;
        }
            
        
        #ifndef NO_ZMQ
            if (send_zmq) {
            zmq::message_t message(sizeof(float)*opt.num_points*2);
            thrust::copy(points_device.begin(), points_device.begin()+opt.num_points, static_cast<float*>(message.data()));
            thrust::copy(points_device.begin()+num_nodes+1, points_device.begin()+num_nodes+1+opt.num_points, static_cast<float*>(message.data())+opt.num_points);
            bool res = false;
            res = publisher.send(message);
            zmq::message_t request;
            res = publisher.recv(&request);
            if (!res) {
                std::cout << "Server Disconnected, Not sending anymore for this session." << std::endl;
            }
            send_zmq = res;
            }
        #endif

        if (opt.get_dump_points() && step % opt.get_dump_interval() == 0) {
            thrust::copy(points_device.begin(), points_device.end(), host_ys);
            for (int i = 0; i < opt.num_points; i++) {
                dump_file << host_ys[i] << " " << host_ys[i + num_nodes + 1] << std::endl;
            }
        }

        // Handle snapshoting
        if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT && step % snap_interval == 0 && opt.return_data != nullptr) {
          thrust::copy(points_device.begin(),
                       points_device.begin()+opt.num_points, 
                       snap_num*opt.num_points*2 + opt.return_data);
          thrust::copy(points_device.begin()+num_nodes+1, 
                       points_device.begin()+num_nodes+1+opt.num_points,
                       snap_num*opt.num_points*2 + opt.return_data+opt.num_points);
          snap_num += 1;
        }

    }

    // Clean up the dump file if we are dumping points
    if (opt.get_dump_points()){
      delete[] host_ys;
      dump_file.close();
    }

    // Handle a once off return type
    if (opt.return_style == tsnecuda::RETURN_STYLE::ONCE && opt.return_data != nullptr) {
      thrust::copy(points_device.begin(), points_device.begin()+opt.num_points, opt.return_data);
      thrust::copy(points_device.begin()+num_nodes+1, points_device.begin()+num_nodes+1+opt.num_points, opt.return_data+opt.num_points);
    }

    // Handle snapshoting
    if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT && opt.return_data != nullptr) {
      thrust::copy(points_device.begin(), points_device.begin()+opt.num_points, snap_num*opt.num_points*2 + opt.return_data);
      thrust::copy(points_device.begin()+num_nodes+1, points_device.begin()+num_nodes+1+opt.num_points, snap_num*opt.num_points*2 + opt.return_data+opt.num_points);
    }

    // Return some final values
    opt.trained = true;
    opt.trained_norm = normalization;

    return;
}

