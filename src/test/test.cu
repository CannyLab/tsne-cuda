/**
 * @brief Unit Tests for the CUDA-TSNE Library
 * 
 * @file test.cu
 * @author David Chan
 * @date 2018-04-04
 */

#include "common.h"
#include "util/cuda_utils.h"
#include "util/random_utils.h"
#include "util/distance_utils.h"
#include "naive_tsne.h"
#include "naive_tsne_cpu.h"
#include <time.h>

// G-Test include
#include "gtest/gtest.h"

// Testing scripts
#include "test/test_distance.h"
#include "test/test_reduce.h"
#include "test/test_tsne.h"

namespace {

    // Pairwise Distance Tests
    TEST(PairwiseDist, 4x2) {test_pairwise_distance(4,2);}
    TEST(PairwiseDist, 64x2) {test_pairwise_distance(64,2);}
    TEST(PairwiseDist, 64x64) {test_pairwise_distance(64,64);}
    TEST(PairwiseDist, 64x128) {test_pairwise_distance(64,128);}

    // Reduction Tests
    TEST(Reductions, ReduceSum_Col_512x512){test_reduce_sum_col(512,512);}
    TEST(Reductions, ReduceSum_Row_512x512){test_reduce_sum_row(512,512);}
    TEST(Reductions, ReduceMean_Col_512x512){test_reduce_mean_col(512,512);}
    TEST(Reductions, ReduceMean_Row_512x512){test_reduce_mean_row(512,512);}
    TEST(Reductions, ReduceAlpha_Col_512x512_pos){test_reduce_alpha_col(512,512,0.1);}
    TEST(Reductions, ReduceAlpha_Row_512x512_pos){test_reduce_alpha_row(512,512,0.1);}
    TEST(Reductions, ReduceAlpha_Col_512x512_neg){test_reduce_alpha_col(512,512,-0.1);}
    TEST(Reductions, ReduceAlpha_Row_512x512_neg){test_reduce_alpha_row(512,512, -0.1);}

    // TEST(ComputePIJ, CPU_256x50) {test_cpu_compute_pij(256, 50);}
    TEST(ComputePIJ, GPUisCPU) {test_cpu_is_gpu_pij(16,16);}

    // T-SNE tests
    TEST(NaiveTSNE, 256x50) {test_tsne(16, 50);}
    
}
