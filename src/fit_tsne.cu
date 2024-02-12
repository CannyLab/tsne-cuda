/*
    Compute t-SNE via Barnes-Hut for NlogN time.
*/

#include "include/fit_tsne.h"
#include <chrono>

#define START_IL_TIMER() start = std::chrono::high_resolution_clock::now();
#define END_IL_TIMER(x)                                                             \
    stop = std::chrono::high_resolution_clock::now();                               \
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
    x += duration;                                                                  \
    total_time += duration;
#define PRINT_IL_TIMER(x) std::cout << #x << ": " << ((float)x.count()) / 1000000.0 << "s" << std::endl

void tsnecuda::RunTsne(tsnecuda::Options &opt,
                       tsnecuda::GpuOptions &gpu_opt)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    auto total_time = duration;
    auto _time_initialization = duration;
    auto _time_knn = duration;
    auto _time_symmetry = duration;
    auto _time_init_low_dim = duration;
    auto _time_init_fft = duration;
    auto _time_precompute_2d = duration;
    auto _time_nbodyfft = duration;
    auto _time_compute_charges = duration;
    auto _time_other = duration;
    auto _time_norm = duration;
    auto _time_attr = duration;
    auto _time_apply_forces = duration;

    // Check the validity of the options file
    if (!opt.validate())
    {
        std::cout << "E: Invalid options file. Terminating." << std::endl;
        return;
    }

    START_IL_TIMER();

    if (opt.verbosity > 0)
    {
        std::cout << "Initializing cuda handles... " << std::flush;
    }

    // Construct the handles
    // TODO: Move this outside of the timing code, since RAPIDs is cheating by pre-initializing the handle.
    // TODO: Allow for multi-stream on the computation, since we can overlap portions of our computation to be quicker.
    cublasHandle_t dense_handle;
    CublasSafeCall(cublasCreate(&dense_handle));
    cusparseHandle_t sparse_handle;
    CusparseSafeCall(cusparseCreate(&sparse_handle));

    // TODO: Pre-allocate device memory, and look for the ability to reuse in our code

    // Set CUDA device properties
    // TODO: Add new GPUs to the gpu_opt, and tune for that.
    const int num_blocks = gpu_opt.sm_count;

    // Construct sparse matrix descriptor
    cusparseMatDescr_t sparse_matrix_descriptor;
    cusparseCreateMatDescr(&sparse_matrix_descriptor);
    cusparseSetMatType(sparse_matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(sparse_matrix_descriptor, CUSPARSE_INDEX_BASE_ZERO);

    // Setup some return information if we're working on snapshots
    // TODO: Add compile flag to remove snapshotting for timing parity
    int snap_num = 0;
    int snap_interval = 1;
    if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT)
    {
        snap_interval = opt.iterations / (opt.num_snapshots - 1);
    }

    // Get constants from options
    const int num_points = opt.num_points;

    // TODO: Warn if the number of neighbors is more than the number of points
    const int num_neighbors = (opt.num_neighbors < num_points) ? opt.num_neighbors : num_points;
    const float *high_dim_points = opt.points;
    const int high_dim = opt.num_dims;
    const float perplexity = opt.perplexity;
    const float perplexity_search_epsilon = opt.perplexity_search_epsilon;
    const float eta = opt.learning_rate;
    float momentum = opt.pre_exaggeration_momentum;
    float attr_exaggeration = opt.early_exaggeration;
    float normalization;

    // Allocate host memory
    // TODO: Pre-determine GPU/CPU memory requirements, since we will know them ahead of time, and can estimate
    // if you're going to run out of GPU memory
    // TODO: Investigate what it takes to use unified memory + Async fetch and execution
    float *knn_squared_distances = new float[num_points * num_neighbors];
    memset(knn_squared_distances, 0, num_points * num_neighbors * sizeof(float));
    int64_t *knn_indices = new int64_t[num_points * num_neighbors];

    // Set cache configs
    // cudaFuncSetCacheConfig(tsnecuda::IntegrationKernel, cudaFuncCachePreferL1);
    // cudaFuncSetCacheConfig(tsnecuda::ComputePijxQijKernel, cudaFuncCachePreferShared);
    GpuErrorCheck(cudaDeviceSynchronize());

    END_IL_TIMER(_time_initialization);
    START_IL_TIMER();

    if (opt.verbosity > 0)
    {
        std::cout << "done.\nKNN Computation... " << std::flush;
    }
    // Compute approximate K Nearest Neighbors and squared distances
    // TODO: See if we can gain some time here by updating FAISS, and building better indicies
    // TODO: Add suport for arbitrary metrics on GPU (Introduced by recent FAISS computation)
    // TODO: Expose Multi-GPU computation (+ Add streaming memory support for GPU optimization)
    tsnecuda::util::KNearestNeighbors(gpu_opt, opt, knn_indices, knn_squared_distances, high_dim_points, high_dim, num_points, num_neighbors);
    thrust::device_vector<int64_t> knn_indices_long_device(knn_indices, knn_indices + num_points * num_neighbors);
    thrust::device_vector<int> pij_indices_device(num_points * num_neighbors);
    tsnecuda::util::PostprocessNeighborIndices(gpu_opt, pij_indices_device, knn_indices_long_device,
                                               num_points, num_neighbors);

    // Max-norm the distances to avoid exponentiating by large numbers
    thrust::device_vector<float> knn_squared_distances_device(knn_squared_distances,
                                                              knn_squared_distances + (num_points * num_neighbors));
    tsnecuda::util::MaxNormalizeDeviceVector(knn_squared_distances_device);

    END_IL_TIMER(_time_knn);
    START_IL_TIMER();

    if (opt.verbosity > 0)
    {
        std::cout << "done.\nComputing Pij matrix... " << std::endl;
    }

    // Search Perplexity
    thrust::device_vector<float> pij_non_symmetric_device(num_points * num_neighbors);
    tsnecuda::SearchPerplexity(gpu_opt, dense_handle, pij_non_symmetric_device, knn_squared_distances_device,
                               perplexity, perplexity_search_epsilon, num_points, num_neighbors);

    // Clean up memory
    cudaDeviceSynchronize();
    knn_squared_distances_device.clear();
    knn_squared_distances_device.shrink_to_fit();
    // knn_indices_long_device.clear();
    // knn_indices_long_device.shrink_to_fit();
    delete[] knn_squared_distances;
    delete[] knn_indices;

    // Symmetrize the pij matrix
    thrust::device_vector<float> pij_device(num_points * num_neighbors);
    tsnecuda::util::SymmetrizeMatrixV2(pij_device, pij_non_symmetric_device, pij_indices_device, num_points, num_neighbors);

    // Clean up memory
    pij_non_symmetric_device.clear();
    pij_non_symmetric_device.shrink_to_fit();

    // Declare memory
    thrust::device_vector<float> pij_workspace_device(num_points * num_neighbors * 2);
    thrust::device_vector<float> repulsive_forces_device(opt.num_points * 2, 0);
    thrust::device_vector<float> attractive_forces_device(opt.num_points * 2, 0);
    thrust::device_vector<float> gains_device(opt.num_points * 2, 1);
    thrust::device_vector<float> old_forces_device(opt.num_points * 2, 0); // for momentum
    thrust::device_vector<float> normalization_vec_device(opt.num_points);
    thrust::device_vector<float> ones_device(opt.num_points * 2, 1); // This is for reduce summing, etc.
    // thrust::device_vector<int> coo_indices_device(sparse_pij_device.size() * 2);

    // tsnecuda::util::Csr2Coo(gpu_opt, coo_indices_device, pij_row_ptr_device,
    //                         pij_col_ind_device, num_points, num_nonzero);

    END_IL_TIMER(_time_symmetry);
    START_IL_TIMER();

    if (opt.verbosity > 0)
    {
        std::cout << "done.\nInitializing low dim points... " << std::flush;
    }

    // Initialize Low-Dim Points
    thrust::device_vector<float> points_device(num_points * 2);
    thrust::device_vector<float> random_vector_device(points_device.size());

    std::default_random_engine generator(opt.random_seed);
    std::normal_distribution<float> distribution1(0.0, 1.0);
    thrust::host_vector<float> h_points_device(num_points * 2);

    // Initialize random noise vector
    for (int i = 0; i < h_points_device.size(); i++)
        h_points_device[i] = 0.001 * distribution1(generator);
    thrust::copy(h_points_device.begin(), h_points_device.end(), random_vector_device.begin());

    // TODO: this will only work with gaussian init
    if (opt.initialization == tsnecuda::TSNE_INIT::UNIFORM)
    { // Random uniform initialization
        points_device = tsnecuda::util::RandomDeviceVectorInRange(generator, points_device.size(), -5, 5);
    }
    else if (opt.initialization == tsnecuda::TSNE_INIT::GAUSSIAN)
    { // Random gaussian initialization
        // Generate some Gaussian noise for the points
        for (int i = 0; i < h_points_device.size(); i++)
            h_points_device[i] = 0.0001 * distribution1(generator);
        thrust::copy(h_points_device.begin(), h_points_device.end(), points_device.begin());
    }
    else if (opt.initialization == tsnecuda::TSNE_INIT::RESUME)
    { // Preinit from vector
        // Load from vector
        if (opt.preinit_data != nullptr)
        {
            thrust::copy(opt.preinit_data, opt.preinit_data + points_device.size(), points_device.begin());
        }
        else
        {
            std::cerr << "E: Invalid initialization. Initialization points are null." << std::endl;
            exit(1);
        }
    }
    else if (opt.initialization == tsnecuda::TSNE_INIT::VECTOR)
    { // Preinit from vector points only
        // Copy the pre-init data
        if (opt.preinit_data != nullptr)
        {
            thrust::copy(opt.preinit_data, opt.preinit_data + points_device.size(), points_device.begin());
        }
        else
        {
            std::cerr << "E: Invalid initialization. Initialization points are null." << std::endl;
            exit(1);
        }
    }
    else
    { // Invalid initialization
        std::cerr << "E: Invalid initialization type specified." << std::endl;
        exit(1);
    }

    END_IL_TIMER(_time_init_low_dim);
    START_IL_TIMER();

    if (opt.verbosity > 0)
    {
        std::cout << "done.\nInitializing CUDA memory... " << std::flush;
    }

    // FIT-TNSE Parameters
    const int n_interpolation_points = 3;
    // float intervals_per_integer = 1;
    int min_num_intervals = 125;
    int N = num_points;
    // int D = 2;
    // The number of "charges" or s+2 sums i.e. number of kernel sums
    int n_terms = 4;
    int n_boxes_per_dim = min_num_intervals;

    // FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
    // 11^e 13^f, where e+f is either 0 or 1, and the other exponents are
    // arbitrary
    int allowed_n_boxes_per_dim[21] = {25, 36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140, 150, 175, 200, 1125};
    if (n_boxes_per_dim < allowed_n_boxes_per_dim[20])
    {
        //Round up to nearest grid point
        int chosen_i;
        for (chosen_i = 0; allowed_n_boxes_per_dim[chosen_i] < n_boxes_per_dim; chosen_i++)
            ;
        n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
    }

    int n_total_boxes = n_boxes_per_dim * n_boxes_per_dim;
    int total_interpolation_points = n_total_boxes * n_interpolation_points * n_interpolation_points;
    int n_fft_coeffs_half = n_interpolation_points * n_boxes_per_dim;
    int n_fft_coeffs = 2 * n_interpolation_points * n_boxes_per_dim;
    int n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;

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
    thrust::device_vector<float> box_lower_bounds_device(2 * n_total_boxes);
    thrust::device_vector<float> box_upper_bounds_device(2 * n_total_boxes);
    thrust::device_vector<float> kernel_tilde_device(n_fft_coeffs * n_fft_coeffs);
    thrust::device_vector<thrust::complex<float>> fft_kernel_tilde_device(2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d);
    thrust::device_vector<float> fft_input(n_terms * n_fft_coeffs * n_fft_coeffs);
    thrust::device_vector<thrust::complex<float>> fft_w_coefficients(n_terms * n_fft_coeffs * (n_fft_coeffs / 2 + 1));
    thrust::device_vector<float> fft_output(n_terms * n_fft_coeffs * n_fft_coeffs);

    // std::cout << "Floats allocated: " << n_terms * n_fft_coeffs * (n_fft_coeffs / 2 + 1) + 2 * n_terms * n_fft_coeffs * n_fft_coeffs + 2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d + n_fft_coeffs * n_fft_coeffs + 4 * n_total_boxes + 2 * N * n_terms + total_interpolation_points * n_terms + 2 * N * n_interpolation_points + total_interpolation_points * n_terms + N + N + N + 4 * n_terms * n_interpolation_points * n_interpolation_points * N << std::endl;

    // Easier to compute denominator on CPU, so we should just calculate y_tilde_spacing on CPU also
    float h = 1 / (float)n_interpolation_points;
    float y_tilde_spacings[n_interpolation_points];
    y_tilde_spacings[0] = h / 2;
    for (int i = 1; i < n_interpolation_points; i++)
    {
        y_tilde_spacings[i] = y_tilde_spacings[i - 1] + h;
    }
    float denominator[n_interpolation_points];
    for (int i = 0; i < n_interpolation_points; i++)
    {
        denominator[i] = 1;
        for (int j = 0; j < n_interpolation_points; j++)
        {
            if (i != j)
            {
                denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
            }
        }
    }
    thrust::device_vector<float> y_tilde_spacings_device(y_tilde_spacings, y_tilde_spacings + n_interpolation_points);
    thrust::device_vector<float> denominator_device(denominator, denominator + n_interpolation_points);

    // Create the FFT Handles
    cufftHandle plan_kernel_tilde, plan_dft, plan_idft;

    CufftSafeCall(cufftCreate(&plan_kernel_tilde));
    CufftSafeCall(cufftCreate(&plan_dft));
    CufftSafeCall(cufftCreate(&plan_idft));

    size_t work_size, work_size_dft, work_size_idft;
    int fft_dimensions[2] = {n_fft_coeffs, n_fft_coeffs};
    CufftSafeCall(cufftMakePlan2d(plan_kernel_tilde, fft_dimensions[0], fft_dimensions[1], CUFFT_R2C, &work_size));
    CufftSafeCall(cufftMakePlanMany(plan_dft, 2, fft_dimensions,
                                    NULL, 1, n_fft_coeffs * n_fft_coeffs,
                                    NULL, 1, n_fft_coeffs * (n_fft_coeffs / 2 + 1),
                                    CUFFT_R2C, n_terms, &work_size_dft));
    CufftSafeCall(cufftMakePlanMany(plan_idft, 2, fft_dimensions,
                                    NULL, 1, n_fft_coeffs * (n_fft_coeffs / 2 + 1),
                                    NULL, 1, n_fft_coeffs * n_fft_coeffs,
                                    CUFFT_C2R, n_terms, &work_size_idft));

    // Dump file
    float *host_ys = nullptr;
    std::ofstream dump_file;
    if (opt.get_dump_points())
    {
        dump_file.open(opt.get_dump_file());
        host_ys = new float[num_points * 2];
        dump_file << num_points << " " << 2 << std::endl;
    }

#ifndef NO_ZMQ
    bool send_zmq = opt.get_use_interactive();
    zmq::context_t context(1);
    zmq::socket_t publisher(context, ZMQ_REQ);
    if (opt.get_use_interactive())
    {

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
        send_zmq = publisher.recv(&request);

        // If there's a time-out, don't bother.
        if (send_zmq)
        {
            if (opt.verbosity >= 1)
                std::cout << "Visualization connected!" << std::endl;
        }
        else
        {
            std::cout << "No Visualization Terminal, continuing..." << std::endl;
            send_zmq = false;
        }
    }
#else
    if (opt.get_use_interactive())
        std::cout << "This version is not built with ZMQ for interative viz. Rebuild with WITH_ZMQ=TRUE for viz." << std::endl;
#endif

    if (opt.verbosity > 0)
    {
        std::cout << "done." << std::endl;
    }

    END_IL_TIMER(_time_init_fft);
    // Support for infinite iteration
    for (size_t step = 0; step != opt.iterations; step++)
    {

        START_IL_TIMER();
        // TODO: We might be able to write a kernel which does this more efficiently. It probably doesn't require much
        // TODO: but it could be done.
        float fill_value = 0;
        thrust::fill(w_coefficients_device.begin(), w_coefficients_device.end(), fill_value);
        thrust::fill(potentialsQij_device.begin(), potentialsQij_device.end(), fill_value);
        // Setup learning rate schedule
        if (step == opt.force_magnify_iters)
        {
            momentum = opt.post_exaggeration_momentum;
            attr_exaggeration = 1.0f;
        }
        END_IL_TIMER(_time_other);

        // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
        START_IL_TIMER();
        tsnecuda::ComputeChargesQij(chargesQij_device, points_device, num_points, n_terms);
        END_IL_TIMER(_time_compute_charges);

        // Compute Minimax elements
        START_IL_TIMER();
        auto minimax_iter = thrust::minmax_element(points_device.begin(), points_device.end());
        float min_coord = minimax_iter.first[0];
        float max_coord = minimax_iter.second[0];

        // Compute the number of boxes in a single dimension and the total number of boxes in 2d
        // auto n_boxes_per_dim = static_cast<int>(fmax(min_num_intervals, (max_coord - min_coord) / intervals_per_integer));

        tsnecuda::PrecomputeFFT2D(
            plan_kernel_tilde, max_coord, min_coord, max_coord, min_coord, n_boxes_per_dim, n_interpolation_points,
            box_lower_bounds_device, box_upper_bounds_device, kernel_tilde_device,
            fft_kernel_tilde_device);

        float box_width = ((max_coord - min_coord) / (float)n_boxes_per_dim);

        END_IL_TIMER(_time_precompute_2d);
        START_IL_TIMER();

        tsnecuda::NbodyFFT2D(
            plan_dft, plan_idft,
            N, n_terms, n_boxes_per_dim, n_interpolation_points,
            fft_kernel_tilde_device, n_total_boxes,
            total_interpolation_points, min_coord, box_width, n_fft_coeffs_half, n_fft_coeffs,
            fft_input, fft_w_coefficients, fft_output,
            point_box_idx_device, x_in_box_device, y_in_box_device, points_device,
            box_lower_bounds_device, y_tilde_spacings_device, denominator_device, y_tilde_values,
            all_interpolated_values_device, output_values, all_interpolated_indices,
            output_indices, w_coefficients_device, chargesQij_device, x_interpolated_values_device,
            y_interpolated_values_device, potentialsQij_device);

        END_IL_TIMER(_time_nbodyfft);
        START_IL_TIMER();

        // TODO: We can overlap the computation of the attractive and repulsive forces, this requires changing the
        // TODO: default streams of the code in both of these methods
        // TODO: See: https://stackoverflow.com/questions/24368197/getting-cuda-thrust-to-use-a-cuda-stream-of-your-choice
        // Make the negative term, or F_rep in the equation 3 of the paper
        normalization = tsnecuda::ComputeRepulsiveForces(
            repulsive_forces_device, normalization_vec_device, points_device,
            potentialsQij_device, num_points, n_terms);

        END_IL_TIMER(_time_norm);
        START_IL_TIMER();

        // Calculate Attractive Forces
        // tsnecuda::ComputeAttractiveForces(gpu_opt,
        //                                   sparse_handle,
        //                                   sparse_matrix_descriptor,
        //                                   attractive_forces_device,
        //                                   sparse_pij_device,
        //                                   pij_row_ptr_device,
        //                                   pij_col_ind_device,
        //                                   coo_indices_device,
        //                                   points_device,
        //                                   ones_device,
        //                                   num_points,
        //                                   num_nonzero);
        tsnecuda::ComputeAttractiveForcesV3(dense_handle,
                                            gpu_opt,
                                            attractive_forces_device,
                                            pij_device,
                                            pij_indices_device,
                                            pij_workspace_device,
                                            points_device,
                                            ones_device,
                                            num_points,
                                            num_neighbors);

        END_IL_TIMER(_time_attr);
        START_IL_TIMER();

        // TODO: Add stream synchronization here.

        // Apply Forces
        tsnecuda::ApplyForces(gpu_opt,
                              points_device,
                              attractive_forces_device,
                              repulsive_forces_device,
                              gains_device,
                              old_forces_device,
                              eta,
                              normalization,
                              momentum,
                              attr_exaggeration,
                              num_points,
                              num_blocks);

        // Compute the gradient norm
        float grad_norm = tsnecuda::util::L2NormDeviceVector(old_forces_device);
        thrust::fill(attractive_forces_device.begin(), attractive_forces_device.end(), 0.0f);

        if (grad_norm < opt.min_gradient_norm)
        {
            if (opt.verbosity >= 1)
                std::cout << "Reached minimum gradient norm: " << grad_norm << std::endl;
            break;
        }

        if (opt.verbosity >= 1 && step % opt.print_interval == 0)
        {
            std::cout << "[Step " << step << "] Avg. Gradient Norm: " << grad_norm << std::endl;
        }

        END_IL_TIMER(_time_apply_forces);

#ifndef NO_ZMQ
        if (send_zmq)
        {
            zmq::message_t message(sizeof(float) * opt.num_points * 2);
            thrust::copy(points_device.begin(), points_device.end(), static_cast<float *>(message.data()));
            bool res = false;
            res = publisher.send(message);
            zmq::message_t request;
            res = publisher.recv(&request);
            if (!res)
            {
                std::cout << "Server Disconnected, Not sending anymore for this session." << std::endl;
            }
            send_zmq = res;
        }
#endif

        if (opt.get_dump_points() && step % opt.get_dump_interval() == 0)
        {
            thrust::copy(points_device.begin(), points_device.end(), host_ys);
            for (int i = 0; i < opt.num_points; i++)
            {
                dump_file << host_ys[i] << " " << host_ys[i + num_points] << std::endl;
            }
        }

        // Handle snapshoting
        if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT && step % snap_interval == 0 && opt.return_data != nullptr)
        {
            thrust::copy(points_device.begin(),
                         points_device.end(),
                         snap_num * opt.num_points * 2 + opt.return_data);
            snap_num += 1;
        }
    } // End for loop

    CufftSafeCall(cufftDestroy(plan_kernel_tilde));
    CufftSafeCall(cufftDestroy(plan_dft));
    CufftSafeCall(cufftDestroy(plan_idft));

    if (opt.verbosity > 0)
    {
        PRINT_IL_TIMER(_time_initialization);
        PRINT_IL_TIMER(_time_knn);
        PRINT_IL_TIMER(_time_symmetry);
        PRINT_IL_TIMER(_time_init_low_dim);
        PRINT_IL_TIMER(_time_init_fft);
        PRINT_IL_TIMER(_time_compute_charges);
        PRINT_IL_TIMER(_time_precompute_2d);
        PRINT_IL_TIMER(_time_nbodyfft);
        PRINT_IL_TIMER(_time_norm);
        PRINT_IL_TIMER(_time_attr);
        PRINT_IL_TIMER(_time_apply_forces);
        PRINT_IL_TIMER(_time_other);
        PRINT_IL_TIMER(total_time);
    }

    // Clean up the dump file if we are dumping points
    if (opt.get_dump_points())
    {
        delete[] host_ys;
        dump_file.close();
    }

    // Handle a once off return type
    if (opt.return_style == tsnecuda::RETURN_STYLE::ONCE && opt.return_data != nullptr)
    {
        thrust::copy(points_device.begin(), points_device.end(), opt.return_data);
    }

    // Handle snapshoting
    if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT && opt.return_data != nullptr)
    {
        thrust::copy(points_device.begin(), points_device.end(), snap_num * opt.num_points * 2 + opt.return_data);
    }

    // Return some final values
    opt.trained = true;
    opt.trained_norm = normalization;

    return;
}
