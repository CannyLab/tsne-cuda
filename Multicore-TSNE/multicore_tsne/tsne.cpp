/*
 *  tsne.cpp
 *  Implementation of both standard and Barnes-Hut-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

// #include "quadtree.h"
#include "splittree.h"
#include "vptree.h"
#include "tsne.h"


#ifdef _OPENMP
    #define NUM_THREADS(N) ((N) >= 0 ? (N) : omp_get_num_procs() + (N) + 1)
#else
    #define NUM_THREADS(N) (1)
#endif


/*  
    Perform t-SNE
        X -- double matrix of size [N, D]
        D -- input dimensionality
        Y -- array to fill with the result of size [N, no_dims]
        no_dims -- target dimentionality
*/
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
void TSNE<treeT, dist_fn>::run(double* X, int N, int D, double* Y,
               int no_dims, double perplexity, double theta ,
               int num_threads, int max_iter, int random_state,
               bool init_from_Y, int verbose,
               double early_exaggeration, double learning_rate,
               double *final_error) {

    if (N - 1 < 3 * perplexity) {
        perplexity = (N - 1) / 3;
        if (verbose)
            fprintf(stderr, "Perplexity too large for the number of data points! Adjusting ...\n");
    }

#ifdef _OPENMP
    omp_set_num_threads(NUM_THREADS(num_threads));
#if _OPENMP >= 200805
    omp_set_schedule(omp_sched_guided, 0);
#endif
#endif

    /* 
        ======================
            Step 1
        ======================
    */

    if (verbose)
        fprintf(stderr, "Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);

    // Set learning parameters
    float total_time = .0;
    time_t start, end;
    int stop_lying_iter = 250, mom_switch_iter = 250;
    double momentum = .5, final_momentum = .8;
    double eta = learning_rate;

    // Allocate some memory
    double* dY    = (double*) malloc(N * no_dims * sizeof(double));
    double* uY    = (double*) calloc(N * no_dims , sizeof(double));
    double* gains = (double*) malloc(N * no_dims * sizeof(double));
    if (dY == NULL || uY == NULL || gains == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int i = 0; i < N * no_dims; i++) {
        gains[i] = 1.0;
    }

    // Normalize input data (to prevent numerical problems)
    if (verbose)
        fprintf(stderr, "Computing input similarities...\n");

    start = time(0);
    zeroMean(X, N, D);
    double max_X = .0;
    for (int i = 0; i < N * D; i++) {
        if (X[i] > max_X) max_X = X[i];
    }
    for (int i = 0; i < N * D; i++) {
        X[i] /= max_X;
    }

    // Compute input similarities
    int* row_P; int* col_P; double* val_P;

    // Compute asymmetric pairwise input similarities
    computeGaussianPerplexity(X, N, D, &row_P, &col_P, &val_P, perplexity, (int) (3 * perplexity), verbose);

    // Symmetrize input similarities
    symmetrizeMatrix(&row_P, &col_P, &val_P, N);
    double sum_P = .0;
    for (int i = 0; i < row_P[N]; i++) {
        sum_P += val_P[i];
    }
    for (int i = 0; i < row_P[N]; i++) {
        val_P[i] /= sum_P;
    }

    end = time(0);
    if (verbose)
        fprintf(stderr, "Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float)(end - start) , (double) row_P[N] / ((double) N * (double) N));

    /* 
        ======================
            Step 2
        ======================
    */


    // Lie about the P-values
    for (int i = 0; i < row_P[N]; i++) {
        val_P[i] *= early_exaggeration;
    }

    // Initialize solution (randomly), unless Y is already initialized
    if (init_from_Y) {
        stop_lying_iter = 0;  // Immediately stop lying. Passed Y is close to the true solution.
    }
    else {
        if (random_state != -1) {
            srand(random_state);
        }
        for (int i = 0; i < N * no_dims; i++) {
            Y[i] = randn();
        }
    }

    // Perform main training loop
    start = time(0);
    for (int iter = 0; iter < max_iter; iter++) {

        bool need_eval_error = (verbose && ((iter > 0 && iter % 50 == 0) || (iter == max_iter - 1)));

        // Compute approximate gradient
        double error = computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta, need_eval_error);

        for (int i = 0; i < N * no_dims; i++) {
            // Update gains
            gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8 + .01);

            // Perform gradient update (with momentum and gains)
            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] = Y[i] + uY[i];
        }

        // Make solution zero-mean
        zeroMean(Y, N, no_dims);

        // Stop lying about the P-values after a while, and switch momentum
        if (iter == stop_lying_iter) {
            for (int i = 0; i < row_P[N]; i++) {
                val_P[i] /= early_exaggeration;
            }
        }
        if (iter == mom_switch_iter) {
            momentum = final_momentum;
        }

        // Print out progress
        if (need_eval_error) {
            end = time(0);

            if (iter == 0)
                fprintf(stderr, "Iteration %d: error is %f\n", iter + 1, error);
            else {
                total_time += (float) (end - start);
                fprintf(stderr, "Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter + 1, error, (float) (end - start) );
            }
            start = time(0);
        }

    }
    end = time(0); total_time += (float) (end - start) ;

    if (final_error != NULL)
        *final_error = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);

    // Clean up memory
    free(dY);
    free(uY);
    free(gains);

    free(row_P); row_P = NULL;
    free(col_P); col_P = NULL;
    free(val_P); val_P = NULL;

    if (verbose)
        fprintf(stderr, "Fitting performed in %4.2f seconds.\n", total_time);
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
double TSNE<treeT, dist_fn>::computeGradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, int no_dims, double* dC, double theta, bool eval_error)
{
    // Construct quadtree on current map
    treeT* tree = new treeT(Y, N, no_dims);
    
    // Compute all terms required for t-SNE gradient
    double* Q = new double[N];
    double* pos_f = new double[N * no_dims]();
    double* neg_f = new double[N * no_dims]();

    double P_i_sum = 0.;
    double C = 0.;

    if (pos_f == NULL || neg_f == NULL) { 
        fprintf(stderr, "Memory allocation failed!\n"); exit(1); 
    }
    
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:P_i_sum,C)
#endif
    for (int n = 0; n < N; n++) {
        // Edge forces
        int ind1 = n * no_dims;
        for (int i = inp_row_P[n]; i < inp_row_P[n + 1]; i++) {

            // Compute pairwise distance and Q-value
            double D = .0;
            int ind2 = inp_col_P[i] * no_dims;
            for (int d = 0; d < no_dims; d++) {
                double t = Y[ind1 + d] - Y[ind2 + d];
                D += t * t;
            }
            
            // Sometimes we want to compute error on the go
            if (eval_error) {
                P_i_sum += inp_val_P[i];
                C += inp_val_P[i] * log((inp_val_P[i] + FLT_MIN) / ((1.0 / (1.0 + D)) + FLT_MIN));
            }

            D = inp_val_P[i] / (1.0 + D);
            // Sum positive force
            for (int d = 0; d < no_dims; d++) {
                pos_f[ind1 + d] += D * (Y[ind1 + d] - Y[ind2 + d]);
            }
        }
        
        // NoneEdge forces
        double this_Q = .0;
        tree->computeNonEdgeForces(n, theta, neg_f + n * no_dims, &this_Q);
        Q[n] = this_Q;
    }
    
    double sum_Q = 0.;
    for (int i = 0; i < N; i++) {
        sum_Q += Q[i];
    }

    // Compute final t-SNE gradient
    for (int i = 0; i < N * no_dims; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }

    delete tree;
    delete[] pos_f;
    delete[] neg_f;
    delete[] Q;

    C += P_i_sum * log(sum_Q);

    return C;
}


// Evaluate t-SNE cost function (approximately)
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
double TSNE<treeT, dist_fn>::evaluateError(int* row_P, int* col_P, double* val_P, double* Y, int N, int no_dims, double theta)
{

    // Get estimate of normalization term
    treeT* tree = new treeT(Y, N, no_dims);

    double* buff = new double[no_dims]();
    double sum_Q = .0;
    for (int n = 0; n < N; n++) {
        tree->computeNonEdgeForces(n, theta, buff, &sum_Q);
    }
    delete tree;
    delete[] buff;
    
    // Loop over all edges to compute t-SNE error
    double C = .0;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:C)
#endif
    for (int n = 0; n < N; n++) {
        int ind1 = n * no_dims;
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {
            double Q = .0;
            int ind2 = col_P[i] * no_dims;
            for (int d = 0; d < no_dims; d++) {
                double b  = Y[ind1 + d] - Y[ind2 + d];
                Q += b * b;
            }
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }
    
    return C;
}

// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
void TSNE<treeT, dist_fn>::computeGaussianPerplexity(double* X, int N, int D, int** _row_P, int** _col_P, double** _val_P, double perplexity, int K, int verbose) {

    if (perplexity > K) fprintf(stderr, "Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (int*)    malloc((N + 1) * sizeof(int));
    *_col_P = (int*)    calloc(N * K, sizeof(int));
    *_val_P = (double*) calloc(N * K, sizeof(double));
    if (*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }

    /*
        row_P -- offsets for `col_P` (i)
        col_P -- K nearest neighbors indices (j)
        val_P -- p_{i | j}
    */

    int* row_P = *_row_P;
    int* col_P = *_col_P;
    double* val_P = *_val_P;

    row_P[0] = 0;
    for (int n = 0; n < N; n++) {
        row_P[n + 1] = row_P[n] + K;
    }

    // Build ball tree on data set
    VpTree<DataPoint, dist_fn>* tree = new VpTree<DataPoint, dist_fn>();
    std::vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for (int n = 0; n < N; n++) {
        obj_X[n] = DataPoint(D, n, X + n * D);
    }
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    if (verbose)
        fprintf(stderr, "Building tree...\n");

    int steps_completed = 0;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int n = 0; n < N; n++)
    {
        std::vector<double> cur_P(K);
        std::vector<DataPoint> indices;
        std::vector<double> distances;

        // Find nearest neighbors
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while (!found && iter < 200) {

            // Compute Gaussian kernel row
            for (int m = 0; m < K; m++) {
                cur_P[m] = exp(-beta * distances[m + 1]);
            }

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for (int m = 0; m < K; m++) {
                sum_P += cur_P[m];
            }
            double H = .0;
            for (int m = 0; m < K; m++) {
                H += beta * (distances[m + 1] * cur_P[m]);
            }
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(perplexity);
            if (Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if (Hdiff > 0) {
                    min_beta = beta;
                    if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for (int m = 0; m < K; m++) {
            cur_P[m] /= sum_P;
        }
        for (int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }

        // Print progress
#ifdef _OPENMP
        #pragma omp atomic
#endif
        ++steps_completed;

        if (verbose && steps_completed % (N / 10) == 0)
        {
#ifdef _OPENMP
            #pragma omp critical
#endif
            fprintf(stderr, " - point %d of %d\n", steps_completed, N);
        }
    }

    // Clean up memory
    obj_X.clear();
    delete tree;
}

template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
void TSNE<treeT, dist_fn>::symmetrizeMatrix(int** _row_P, int** _col_P, double** _val_P, int N) {

    // Get sparse matrix
    int* row_P = *_row_P;
    int* col_P = *_col_P;
    double* val_P = *_val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = (int*) calloc(N, sizeof(int));
    if (row_counts == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int n = 0; n < N; n++) {
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n) {
                    present = true;
                    break;
                }
            }
            if (present) {
                row_counts[n]++;
            }
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for (int n = 0; n < N; n++) {
        no_elem += row_counts[n];
    }
    // Allocate memory for symmetrized matrix
    int*    sym_row_P = (int*)    malloc((N + 1) * sizeof(int));
    int*    sym_col_P = (int*)    malloc(no_elem * sizeof(int));
    double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
    if (sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for (int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + row_counts[n];

    // Fill the result matrix
    int* offset = (int*) calloc(N, sizeof(int));
    if (offset == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int n = 0; n < N; n++) {
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {                                 // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n) {
                    present = true;
                    if (n <= col_P[i]) {                                                // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if (!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if (!present || (n <= col_P[i])) {
                offset[n]++;
                if (col_P[i] != n) {
                    offset[col_P[i]]++;
                }
            }
        }
    }

    // Divide the result by two
    for (int i = 0; i < no_elem; i++) {
        sym_val_P[i] /= 2.0;
    }

    // Return symmetrized matrices
    free(*_row_P); *_row_P = sym_row_P;
    free(*_col_P); *_col_P = sym_col_P;
    free(*_val_P); *_val_P = sym_val_P;

    // Free up some memery
    free(offset); offset = NULL;
    free(row_counts); row_counts  = NULL;
}


// Makes data zero-mean
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
void TSNE<treeT, dist_fn>::zeroMean(double* X, int N, int D) {

    // Compute data mean
    double* mean = (double*) calloc(D, sizeof(double));
    if (mean == NULL) { fprintf(stderr, "Memory allocation failed!\n"); exit(1); }
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            mean[d] += X[n * D + d];
        }
    }
    for (int d = 0; d < D; d++) {
        mean[d] /= (double) N;
    }

    // Subtract data mean
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            X[n * D + d] -= mean[d];
        }
    }
    free(mean); mean = NULL;
}


// Generates a Gaussian random number
template <class treeT, double (*dist_fn)( const DataPoint&, const DataPoint&)>
double TSNE<treeT, dist_fn>::randn() {
    double x, radius;
    do {
        x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        double y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while ((radius >= 1.0) || (radius == 0.0));
    radius = sqrt(-2 * log(radius) / radius);
    x *= radius;
    return x;
}

extern "C"
{
    #ifdef _WIN32
    __declspec(dllexport)
    #endif
    extern void tsne_run_double(double* X, int N, int D, double* Y,
                                int no_dims = 2, double perplexity = 30, double theta = .5,
                                int num_threads = 1, int max_iter = 1000, int random_state = -1,
                                bool init_from_Y = false, int verbose = 0,
                                double early_exaggeration = 12, double learning_rate = 200,
                                double *final_error = NULL, int distance = 1)
    {
        if (verbose)
            fprintf(stderr, "Performing t-SNE using %d cores.\n", NUM_THREADS(num_threads));
        if (distance == 0) {
            TSNE<SplitTree, euclidean_distance> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, verbose, early_exaggeration, learning_rate, final_error);
        }
        else {
            TSNE<SplitTree, euclidean_distance_squared> tsne;
            tsne.run(X, N, D, Y, no_dims, perplexity, theta, num_threads, max_iter, random_state,
                     init_from_Y, verbose, early_exaggeration, learning_rate, final_error);
        }
    }
}
