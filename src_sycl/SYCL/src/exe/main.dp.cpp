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

// This file exposes a main file which does most of the testing with command line
// args, so we don't have to re-build to change options.

// Detailed includes
#include <sycl/sycl.hpp>
#include <time.h>
#include <string>
#include "include/fit_tsne.h"
#include "include/options.h"

// Option parser
#include "include/cxxopts.hpp"

#define TIMER_START() time_start = std::chrono::steady_clock::now();
#define TIMER_END()                                                                         \
    time_end = std::chrono::steady_clock::now();                                            \
    time_total  = std::chrono::duration<double, std::milli>(time_end - time_start).count();
#define TIMER_PRINT(name) std::cout << name <<": " << (time_total - time_total_) / 1e3 << " s\n";

// #ifndef DEBUG_TIME
// #define DEBUG_TIME
// #endif

#define STRINGIFY(X) #X

#define FOPT(x) result[STRINGIFY(x)].as<float>()
#define SOPT(x) result[STRINGIFY(x)].as<std::string>()
#define IOPT(x) result[STRINGIFY(x)].as<int>()
#define BOPT(x) result[STRINGIFY(x)].as<bool>()

int main(int argc, char** argv)
{
    std::chrono::steady_clock::time_point time_start;
    std::chrono::steady_clock::time_point time_end;
    double time_total = 0.0;
    double time_total_ = 0.0;

    TIMER_START()

    try {
    // Setup command line options
    cxxopts::Options options("TSNE-CUDA","Perform T-SNE in an optimized manner.");
    options.add_options()
        ("l,learning-rate",     "Learning Rate",                                        cxxopts::value<float>()->default_value("200"))
        ("p,perplexity",        "Perplexity",                                           cxxopts::value<float>()->default_value("50.0"))
        ("e,early-ex",          "Early Exaggeration Factor",                            cxxopts::value<float>()->default_value("12.0"))
        ("s,data",              "Which program to run on <cifar10,cifar100,mnist,sim>", cxxopts::value<std::string>()->default_value("sim"))
        ("k,num-points",        "How many simulated points to use",                     cxxopts::value<int>()->default_value("60000"))
        ("u,nearest-neighbors", "How many nearest neighbors should we use",             cxxopts::value<int>()->default_value("32"))
        ("n,num-steps",         "How many steps to take",                               cxxopts::value<int>()->default_value("1000"))
        ("i,viz",               "Use interactive visualization",                        cxxopts::value<bool>()->default_value("false"))
        ("d,dump",              "Dump the output points",                               cxxopts::value<bool>()->default_value("false"))
        ("m,magnitude-factor",  "Magnitude factor for KNN",                             cxxopts::value<float>()->default_value("5.0"))
        ("t,init",              "What kind of initialization to use <unif,gauss>",      cxxopts::value<std::string>()->default_value("gauss"))
        ("f,fname",             "File name for loaded data...",                         cxxopts::value<std::string>()->default_value("../train-images.idx3-ubyte"))
        ("c,connection",        "Address for connection to vis server",                 cxxopts::value<std::string>()->default_value("tcp://localhost:5556"))
        ("q,dim",               "Point Dimensions",                                     cxxopts::value<int>()->default_value("50"))
        ("j,device",            "Device to run on",                                     cxxopts::value<int>()->default_value("0"))
        ("h,help",              "Print help");

    // Parse command line options
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }

    tsnecuda::TSNE_INIT init_type = tsnecuda::TSNE_INIT::UNIFORM;
    if (SOPT(init).compare("unif") == 0) {
        init_type = tsnecuda::TSNE_INIT::UNIFORM;
    } else {
        init_type = tsnecuda::TSNE_INIT::GAUSSIAN;
    }

    // Do the T-SNE
    printf("Starting TSNE calculation with %u points.\n", IOPT(num-points));

    // Construct the options
    tsnecuda::Options opt(nullptr, IOPT(num-points), IOPT(dim), nullptr);
    opt.perplexity              = FOPT(perplexity);
    opt.learning_rate           = FOPT(learning-rate);
    opt.early_exaggeration      = FOPT(early-ex);
    opt.iterations              = IOPT(num-steps);
    opt.iterations_no_progress  = IOPT(num-steps);
    opt.magnitude_factor        = FOPT(magnitude-factor);
    opt.num_neighbors           = IOPT(nearest-neighbors);
    opt.initialization          = init_type;

    if (BOPT(dump)) {
        opt.enable_dump("dump_ys.txt", 1);
    }
    if (BOPT(viz)) {
        opt.enable_viz(SOPT(connection));
    }

    // Do the t-SNE
    time_total_ = tsnecuda::RunTsne(opt);
    std::cout << "\nDone!\n";
    } catch (std::exception const& e) {
        std::cout << "Exception: " << e.what() << "\n";
    }

    TIMER_END()
    TIMER_PRINT("tsne - total time for whole calculation")

    return 0;
}
