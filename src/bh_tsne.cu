/*
    Compute t-SNE via Barnes-Hut for NlogN time.
*/

#include "bh_tsne.h"
#include "FIt-SNE/src/nbodyfft.h"

float squared_cauchy_2d(float x1, float x2, float y1, float y2) {
    return pow(1.0 + pow(x1 - y1, 2) + pow(x2 - y2, 2), -2);
}
// compute minimum and maximum values in a single reduction

// minmax_pair stores the minimum and maximum 
// values that have been encountered so far
template <typename T>
struct minmax_pair
{
  T min_val;
  T max_val;
};

// minmax_unary_op is a functor that takes in a value x and
// returns a minmax_pair whose minimum and maximum values
// are initialized to x.
template <typename T>
struct minmax_unary_op
  : public thrust::unary_function< T, minmax_pair<T> >
{
  __host__ __device__
  minmax_pair<T> operator()(const T& x) const
  {
    minmax_pair<T> result;
    result.min_val = x;
    result.max_val = x;
    return result;
  }
};

// minmax_binary_op is a functor that accepts two minmax_pair 
// structs and returns a new minmax_pair whose minimum and 
// maximum values are the min() and max() respectively of 
// the minimums and maximums of the input pairs
template <typename T>
struct minmax_binary_op
  : public thrust::binary_function< minmax_pair<T>, minmax_pair<T>, minmax_pair<T> >
{
  __host__ __device__
  minmax_pair<T> operator()(const minmax_pair<T>& x, const minmax_pair<T>& y) const
  {
    minmax_pair<T> result;
    result.min_val = thrust::min(x.min_val, y.min_val);
    result.max_val = thrust::max(x.max_val, y.max_val);
    return result;
  }
};

__global__ void compute_repulsive_forces_kernel(
    volatile float * __restrict__ repulsive_forces_device,
    volatile float * __restrict__ normalization_vec_device,
    const float * const xs,
    const float * const ys,
    const float * const potentialsQij,
    const int num_points,
    const int num_nodes,
    const int n_terms)
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_points)
        return;

    register float phi1, phi2, phi3, phi4, x_pt, y_pt;

    phi1 = potentialsQij[TID * n_terms + 0];
    phi2 = potentialsQij[TID * n_terms + 1];
    phi3 = potentialsQij[TID * n_terms + 2];
    phi4 = potentialsQij[TID * n_terms + 3];

    x_pt = xs[TID];
    y_pt = ys[TID];

    normalization_vec_device[TID] = 
        (1 + x_pt * x_pt + y_pt * y_pt) * phi1 - 2 * (x_pt * phi2 + y_pt * phi3) + phi4;

    repulsive_forces_device[TID] = x_pt * phi1 - phi2;
    repulsive_forces_device[TID + num_nodes + 1] = y_pt * phi1 - phi3;
}

float compute_repulsive_forces(
    thrust::device_vector<float> &repulsive_forces_device,
    thrust::device_vector<float> &normalization_vec_device,
    thrust::device_vector<float> &xs_device,
    thrust::device_vector<float> &ys_device,
    thrust::device_vector<float> &potentialsQij,
    const int num_points,
    const int num_nodes,
    const int n_terms)
{
    const int num_threads = 1024;
    const int num_blocks = (num_points + num_threads - 1) / num_threads;
    compute_repulsive_forces_kernel<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(repulsive_forces_device.data()),
        thrust::raw_pointer_cast(normalization_vec_device.data()),
        thrust::raw_pointer_cast(xs_device.data()),
        thrust::raw_pointer_cast(ys_device.data()),
        thrust::raw_pointer_cast(potentialsQij.data()),
        num_points, num_nodes, n_terms);
    float sumQ = thrust::reduce(
        normalization_vec_device.begin(), normalization_vec_device.end(), 0, 
        thrust::plus<float>());
    return sumQ - num_points;
}

__global__ void compute_chargesQij_kernel(
    volatile float * __restrict__ chargesQij,
    const float * const xs,
    const float * const ys,
    const int num_points,
    const int n_terms)
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_points)
        return;

    register float x_pt, y_pt;
    x_pt = xs[TID];
    y_pt = ys[TID];

    chargesQij[TID * n_terms + 0] = 1;
    chargesQij[TID * n_terms + 1] = x_pt;
    chargesQij[TID * n_terms + 2] = y_pt;
    chargesQij[TID * n_terms + 3] = x_pt * x_pt + y_pt * y_pt;
}

void compute_chargesQij(
    thrust::device_vector<float> &chargesQij,
    thrust::device_vector<float> &xs_device,
    thrust::device_vector<float> &ys_device,
    const int num_points,
    const int n_terms)
{
    const int num_threads = 1024;
    const int num_blocks = (num_points + num_threads - 1) / num_threads;
    compute_chargesQij_kernel<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(chargesQij.data()),
        thrust::raw_pointer_cast(xs_device.data()),
        thrust::raw_pointer_cast(ys_device.data()),
        num_points, n_terms);
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
    // thrust::device_vector<float> normalization_vec_device(num_nodes + 1);
    thrust::device_vector<float> normalization_vec_device(opt.num_points);
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
    
    // FIT-TNSE Parameters
    int n_interpolation_points = 3;
    float intervals_per_integer = 1;
    int min_num_intervals = 50;
    int N = num_points;
    int D = 2;
    // The number of "charges" or s+2 sums i.e. number of kernel sums
    int n_terms = 4;
    int n_boxes_per_dim = min_num_intervals;
    
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

    int n_total_boxes = n_boxes_per_dim * n_boxes_per_dim;
    int total_interpolation_points = n_total_boxes * n_interpolation_points * n_interpolation_points;
    int n_fft_coeffs_half = n_interpolation_points * n_boxes_per_dim;
    int n_fft_coeffs = 2 * n_interpolation_points * n_boxes_per_dim;

    // FIT-TSNE Device Vectors
    thrust::device_vector<int> point_box_idx_device(N);
    thrust::device_vector<float> x_in_box_device(N);
    thrust::device_vector<float> y_in_box_device(N);
    thrust::device_vector<float> y_tilde_values(total_interpolation_points * n_terms);
    thrust::device_vector<float> x_interpolated_values_device(N * n_interpolation_points);
    thrust::device_vector<float> y_interpolated_values_device(N * n_interpolation_points);
    thrust::device_vector<float> potentialsQij_device(N * n_terms);
    thrust::device_vector<float> w_coefficients_device(total_interpolation_points * n_terms);
    thrust::device_vector<float> all_interpolated_values_device(
        n_terms * n_interpolation_points * n_interpolation_points * N);
    thrust::device_vector<float> output_values(
        n_terms * n_interpolation_points * n_interpolation_points * N);
    thrust::device_vector<int> all_interpolated_indices(
        n_terms * n_interpolation_points * n_interpolation_points * N);
    thrust::device_vector<int> output_indices(
        n_terms * n_interpolation_points * n_interpolation_points * N);
    thrust::device_vector<float> chargesQij_device(N * n_terms);
        
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

    float _time_allocate_memory = 0.0f;
    float _time_precompute_2d = 0.0f;
    float _time_nbodyfft = 0.0f;
    float _time_compute_charges = 0.0f;
    float _time_other = 0.0f;
    float _time_norm = 0.0f;
    clock_t _clock = clock();

    #define time(x) x += ( (float) clock() - _clock ) / CLOCKS_PER_SEC; _clock = clock();

    // Support for infinite iteration
    for (size_t step = 0; step != opt.iterations; step++) {
        float fill_value = 0; 
        thrust::fill(w_coefficients_device.begin(), w_coefficients_device.end(), fill_value);
        // Setup learning rate schedule
        if (step == opt.force_magnify_iters) {
            momentum = opt.post_exaggeration_momentum;
            attr_exaggeration = 1.0f;
        }

        // Copy data back from GPU to CPU
        time(_time_other)
	thrust::copy(points_device.begin(), points_device.end(), h_points_device.begin());
        thrust::device_vector<float> xs_device(points_device.begin(), points_device.begin() + num_points);
        thrust::device_vector<float> ys_device(points_device.begin() + num_nodes + 1, points_device.begin() + num_nodes + 1 + num_points);
        minmax_unary_op<float>  unary_op;
        minmax_binary_op<float> binary_op;

        auto xs_minmax = thrust::transform_reduce(
            xs_device.begin(), xs_device.end(), unary_op, unary_op(xs_device[0]), binary_op);
        auto ys_minmax = thrust::transform_reduce(
            ys_device.begin(), ys_device.end(), unary_op, unary_op(ys_device[0]), binary_op);
        float min_coord = fmin(xs_minmax.min_val, ys_minmax.min_val);
        float max_coord = fmax(xs_minmax.max_val, ys_minmax.max_val);

	time(_time_allocate_memory)

        // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
        compute_chargesQij(chargesQij_device, xs_device, ys_device, num_points, n_terms);

	time(_time_compute_charges)
        // Compute the number of boxes in a single dimension and the total number of boxes in 2d
        // auto n_boxes_per_dim = static_cast<int>(fmax(min_num_intervals, (max_coord - min_coord) / intervals_per_integer));

        auto *box_lower_bounds = new float[2 * n_total_boxes];
        auto *box_upper_bounds = new float[2 * n_total_boxes];
        auto *y_tilde_spacings = new float[n_interpolation_points];
        int n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;
        auto *x_tilde = new float[n_interpolation_points_1d]();
        auto *y_tilde = new float[n_interpolation_points_1d]();
        auto *fft_kernel_tilde = new complex<float>[2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d];

	time(_time_allocate_memory)
        precompute_2d(
            max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim, n_interpolation_points,
            &squared_cauchy_2d, box_lower_bounds, box_upper_bounds, y_tilde_spacings, x_tilde, 
            y_tilde, fft_kernel_tilde);

        float denominator[n_interpolation_points];
        for (int i = 0; i < n_interpolation_points; i++) {
            denominator[i] = 1;
            for (int j = 0; j < n_interpolation_points; j++) {
                if (i != j) {
                    denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
                }
            }
        }
        float coord_min = box_lower_bounds[0];
        float box_width = box_upper_bounds[0] - box_lower_bounds[0];
        

        thrust::device_vector<float> box_lower_bounds_device(
            box_lower_bounds, box_lower_bounds + 2 * n_total_boxes);
        thrust::device_vector<float> y_tilde_spacings_device(
            y_tilde_spacings, y_tilde_spacings + n_interpolation_points);
        thrust::device_vector<float> denominator_device(
            denominator, denominator + n_interpolation_points);
        // thrust::device_vector<float> chargesQij_device(
            // chargesQij, chargesQij + N * n_terms);
	time(_time_precompute_2d)

        n_body_fft_2d(
            N, n_terms, n_boxes_per_dim, n_interpolation_points, 
            fft_kernel_tilde, denominator, n_total_boxes, 
            total_interpolation_points, coord_min, box_width, n_fft_coeffs_half, n_fft_coeffs, 
            point_box_idx_device, x_in_box_device, y_in_box_device, xs_device, ys_device, 
            box_lower_bounds_device, y_tilde_spacings_device, denominator_device, y_tilde_values,
            all_interpolated_values_device, output_values, all_interpolated_indices,
            output_indices, w_coefficients_device, chargesQij_device, x_interpolated_values_device,
            y_interpolated_values_device, potentialsQij_device);

	time(_time_nbodyfft)
        // Compute the normalization constant Z or sum of q_{ij}. This expression is different from the one in the original
        // paper, but equivalent. This is done so we need only use a single kernel (K_2 in the paper) instead of two
        // different ones. We subtract N at the end because the following sums over all i, j, whereas Z contains i \neq j
        
        // Make the negative term, or F_rep in the equation 3 of the paper
        normalization = compute_repulsive_forces(
            repulsive_forces_device, normalization_vec_device, xs_device, ys_device, 
            potentialsQij_device, num_points, num_nodes, n_terms);

	time(_time_norm)
        // Copy data back to the GPU

        delete[] box_lower_bounds;
        delete[] box_upper_bounds;
        delete[] y_tilde_spacings;
        delete[] y_tilde;
        delete[] x_tilde;
        delete[] fft_kernel_tilde;
        
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
        float grad_norm = thrust::reduce(
            attractive_forces_device.begin(), attractive_forces_device.begin() + num_points, 
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

    float* nt = get_ntime();

    std::cout << "Time Compute Attractive Forces & Integrate: " << _time_other << "s" << std::endl;
    std::cout << "Time N-Body FFT: " << _time_nbodyfft << "s" << std::endl;
    std::cout << "\t Allocate: " << nt[1] << "s" << std::endl;
    std::cout << "\t Point Box IDX: " << nt[2] << "s" << std::endl;
    std::cout << "\t Interpolate: " << nt[3] << "s" << std::endl;
    std::cout << "\t Reduce: " << nt[4] << "s" << std::endl;
    std::cout << "\t FFT - Init: " << nt[8] << "s" << std::endl;
    std::cout << "\t FFT - Copy To: " << nt[7] << "s" << std::endl;
    std::cout << "\t FFT - Execute: " << nt[9] << "s" << std::endl;
    std::cout << "\t FFT - Copy Back: " << nt[5] << "s" << std::endl;
    std::cout << "\t Potentials: " << nt[6] << "s" << std::endl;
    std::cout << "Time Precompute 2D: " << _time_precompute_2d << "s" << std::endl;
    std::cout << "Time Allocate Memory: " << _time_allocate_memory << "s" << std::endl;
    std::cout << "Time Norm: " << _time_norm << "s" << std::endl;
    std::cout << "Time Compute Charges: " << _time_compute_charges << "s" << std::endl;
    std::cout << "Total Time: " << _time_other + _time_precompute_2d + _time_nbodyfft + _time_norm + _time_precompute_2d + _time_compute_charges << "s" << std::endl;

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

