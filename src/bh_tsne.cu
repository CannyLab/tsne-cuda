// TODO: add copyright

/*
    Compute t-SNE via Barnes-Hut for NlogN time.
*/

#include "bh_tsne.h"

#define WARPSIZE 32

void tsnecuda::bh::RunTsne(cublasHandle_t &dense_handle, 
                            cusparseHandle_t &sparse_handle, 
                            tsnecuda::Options &opt)
{
    // Check the validity of the options file
    if (!opt.validate()) {
        std::cout << "E: Invalid options file. Terminating." << std::endl;
        return;
    }

    // Set CUDA device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.warpSize != WARPSIZE) {
        fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
        exit(-1);
    }
    const uint32_t num_blocks = deviceProp.multiProcessorCount;

    // Get constants from options
    const uint32_t num_points = opt.num_points;
    const uint32_t num_near_neighbors = (opt.num_near_neighbors < num_points) ? opt.num_near_neighbors : num_points;
    const float *high_dim_points = opt.points;
    const uint32_t high_dim = opt.num_dims;
    const float perplexity = opt.perplexity;
    const float perplexity_search_epsilon = opt.perplexity_search_epsilon;
    const float eta = opt.learning_rate;
    float momentum = opt.pre_exaggeration_momentum;
    float attr_exaggeration = opt.early_exaggeration;
    float norm;

    // Theta governs tolerance for Barnes-Hut recursion
    const float theta = opt.theta;
    const float epsilon_squared = opt.epsilon_squared;

    // Figure out number of nodes needed for BH tree
    uint32_t num_nodes = num_points * 2;
    if (num_nodes < 1024 * num_blocks) num_nodes = 1024 * num_blocks;
    while ((num_nodes & (WARPSIZE - 1)) != 0)
        num_nodes++;
    num_nodes--;
    const uint32_t num_nodes = num_nodes;
    opt.num_nodes = num_nodes;
    
    // Allocate host memory
    float *knn_squared_distances = new float[num_points * num_near_neighbors];
    memset(knn_squared_distances, 0, num_points * num_near_neighbors * sizeof(float));
    long *knn_indices = new long[num_points * num_near_neighbors];

    // Initialize global variables
    thrust::device_vector<int> err_device(1);
    tsnecuda::bh::Iniitalize(err_device);

    // Compute approximate K Nearest Neighbors and squared distances
    tsnecuda::util::KNearestNeighbors(knn_indices, knn_squared_distances, high_dim_points, num_near_neighbors);
    thrust::device_vector<long> knn_indices_long_device(knn_indices, knn_indices + num_points * num_near_neighbors);
    thrust::device_vector<int> knn_indices_device(num_points * num_near_neighbors);
    tsnecuda::util::PostprocessNeighborIndices(knn_indices_device, knn_indices_long_device, 
                                                        num_points, num_near_neighbors);
    
    // Max-norm the distances to avoid exponentiating by large numbers
    thrust::device_vector<float> knn_squared_distances_device(knn_distances, 
                                            knn_distances + (num_points * num_near_neighbors));
    tsnecuda::util::MaxNormalizeDeviceVector(knn_squared_distances_device);

    // Search Perplexity
    thrust::device_vector<float> pij_non_symmetric_device(num_points * num_near_neighbors);
    tsnecuda::bh::SearchPerplexity(dense_handle, pij_non_symmetric_device, knn_squared_distances_device, 
                                    perplexity, perplexity_search_epsilon, num_points, num_near_neighbors);

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
    // TODO: fix order of arguments
    tsnecuda::util::SymmetrizeMatrix(sparse_handle, sparse_pij_device, pij_row_ptr_device,
                                        pij_col_ind_device, pij_non_symmetric_device, knn_indices_device,
                                        opt.magnitude_factor, num_points, num_near_neighbors);

    // Clean up memory
    knn_indices_device.clear();
    knn_indices_device.shrink_to_fit();
    pij_non_symmetric_device.clear();
    pij_non_symmetric_device.shrink_to_fit();

    // TODO: compute coo_indices;

    // Initialize Low-Dim Points
    thrust::device_vector<float> points_device((num_nodes + 1) * 2);
    thrust::device_vector<float> random_vector_device(points_device.size());
    // TODO: this will only work with gaussian init
    if (opt.initialization == BHTSNE::TSNE_INIT::UNIFORM) { // Random uniform initialization
        points_device = tsnecuda::util::RandomDeviceVectorInRange((num_nodes+1)*2, -100, 100);
    } else if (opt.initialization == BHTSNE::TSNE_INIT::GAUSSIAN) { // Random gaussian initialization
        std::default_random_engine generator;
        std::normal_distribution<double> distribution1(0.0, 1.0);
        thrust::host_vector<float> h_points_device((num_nodes+ 1) * 2);
        for (int i = 0; i < (num_nodes+ 1) * 2; i++) h_points_device[i] = 0.0001 * distribution1(generator);
        thrust::copy(h_points_device.begin(), h_points_device.end(), points_device.begin());
        thrust::constant_iterator<float> mult(10);
        thrust::transform(points_device.begin(), points_device.end(), mult, random_vector_device.begin(), thrust::multiplies<float>());
    } else if (opt.initialization == BHTSNE::TSNE_INIT::RESUME) { // Preinit from vector
        // Load from vector
        if(opt.preinit_data != nullptr) {
          thrust::copy(opt.preinit_data, opt.preinit_data+(num_nodes+1)*2, points_device.begin());
        } else {
          std::cerr << "E: Invalid initialization. Initialization points are null." << std::endl;
          exit(1);
        }
    } else if (opt.initialization == BHTSNE::TSNE_INIT::VECTOR) { // Preinit from vector points only
        // Load only the points into the pre-init vector
        points_device = tsnecuda::util::RandomDeviceVectorInRange((num_nodes+1)*2, -100, 100);
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
#endif
#ifdef NO_ZMQ
    if (opt.get_use_interactive()) 
        std::cout << "This version is not built with ZMQ for interative viz. Rebuild with WITH_ZMQ=TRUE for viz." << std::endl;
#endif
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
    thrust::device_vector<float> x_max_device(blocks * FACTOR1);
    thrust::device_vector<float> y_max_device(blocks * FACTOR1);
    thrust::device_vector<float> x_min_device(blocks * FACTOR1);
    thrust::device_vector<float> y_min_device(blocks * FACTOR1);
    thrust::device_vector<float> ones_device(opt.num_points * 2, 1); // This is for reduce summing, etc.
    thrust::device_vector<int> coo_indices_device(sparse_pij_device.size()*2);

    // Support for infinite iteration
    for (size_t step = 0; step != opt.iterations; step++) {

        // Setup learning rate schedule
        if (step == opt.force_magnify_iters) {
            momentum = opt.post_exaggeration_momentum;
            attr_exaggeration = 1.0f;
        }

        // Compute Bounding Box
        tsnecuda::bh::ComputeBoundingBox(cell_starts_device,
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
        tsnecuda::bh::BuildTree(err_device,
                                children_device,
                                cell_starts_device,
                                points_device,
                                num_nodes,
                                num_points,
                                num_blocks);

        // Tree Summarization
        tsnecuda::bh::SummarizeTree(cell_counts_device,
                                    children_device,
                                    cell_mass_device,
                                    points_device,
                                    num_nodes,
                                    num_points,
                                    num_blocks);

        // Sort By Morton Code
        tsnecuda::bh::SortCells(cell_sorted_device,
                                cell_starts_device,
                                children_device,
                                cell_counts_device,
                                num_nodes,
                                num_points,
                                num_blocks);

        // Calculate Repulsive Forces
        tsnecuda::bh::ComputeRepulsiveForces(err_device,
                                             repulsive_forces_device,
                                             normalization_vec_device,
                                             cell_sorted_device,
                                             children_device,
                                             cell_mass_device,
                                             points_device,
                                             theta, 
                                             epsilon,
                                             num_nodes,
                                             num_points,
                                             num_blocks);
        // Compute normalization
        normalization = thrust::reduce(normalization_vec_device.begin(),
                                        normalization_vec_device.end(),
                                        0.0f,
                                        thrust::plus<float>());

        // Calculate Attractive Forces
        tsnecuda::bh::ComputeAttractiveForces(sparse_handle,
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
        tsnecuda::bh::ApplyForces(points_device,
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
        if (opt.return_style == BHTSNE::RETURN_STYLE::SNAPSHOT && step % snap_interval == 0 && opt.return_data != nullptr) {
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

    // With verbosity 2, print the timing data
    if (opt.verbosity >= 2) {
      int p1_time = times[0] + times[1] + times[2] + times[3];
      int p2_time = times[4] + times[5] + times[6] + times[7] + times[8] + times[9] + times[10] + times[11] + times[12];
      std::cout << "Timing data: " << std::endl;
      std::cout << "\t Phase 1 (" << p1_time  << "us):" << std::endl;
      std::cout << "\t\tKernel Setup: " << times[0] << "us" << std::endl;
      std::cout << "\t\tKNN Computation: " << times[1] << "us" << std::endl;
      std::cout << "\t\tPIJ Computation: " << times[2] << "us" << std::endl;
      std::cout << "\t\tPIJ Symmetrization: " << times[3] << "us" << std::endl;
      std::cout << "\t Phase 2 (" << p2_time << "us):" << std::endl;
      std::cout << "\t\tKernel Setup: " << times[4] << "us" << std::endl;
      std::cout << "\t\tForce Reset: " << times[5] << "us" << std::endl;
      std::cout << "\t\tBounding Box: " << times[6] << "us" << std::endl;
      std::cout << "\t\tTree Building: " << times[7] << "us" << std::endl;
      std::cout << "\t\tTree Summarization: " << times[8] << "us" << std::endl;
      std::cout << "\t\tSorting: " << times[9] << "us" << std::endl;
      std::cout << "\t\tRepulsive Force Calculation: " << times[10] << "us" << std::endl;
      std::cout << "\t\tAttractive Force Calculation: " << times[11] << "us" << std::endl;
      std::cout << "\t\tIntegration: " << times[12] << "us" << std::endl;
      std::cout << "Total Time: " << p1_time + p2_time << "us" << std::endl << std::endl;
    }

    if (opt.verbosity >= 1) std::cout << "Fin." << std::endl;
    
    // Handle a once off return type
    if (opt.return_style == BHTSNE::RETURN_STYLE::ONCE && opt.return_data != nullptr) {
      thrust::copy(points_device.begin(), points_device.begin()+opt.num_points, opt.return_data);
      thrust::copy(points_device.begin()+num_nodes+1, points_device.begin()+num_nodes+1+opt.num_points, opt.return_data+opt.num_points);
    }

    // Handle snapshoting
    if (opt.return_style == BHTSNE::RETURN_STYLE::SNAPSHOT && opt.return_data != nullptr) {
      thrust::copy(points_device.begin(), points_device.begin()+opt.num_points, snap_num*opt.num_points*2 + opt.return_data);
      thrust::copy(points_device.begin()+num_nodes+1, points_device.begin()+num_nodes+1+opt.num_points, snap_num*opt.num_points*2 + opt.return_data+opt.num_points);
    }

    // Return some final values
    opt.trained = true;
    opt.trained_norm = norm;

    return;
}

