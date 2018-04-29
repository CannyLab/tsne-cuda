/**
 * @brief Unit Tests for the CUDA-TSNE Library
 * 
 * @file test.cu
 * @author David Chan
 * @date 2018-04-04
 */

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

// G-Test include
#include "gtest/gtest.h"

// Testing scripts
#include "test/test_distance.h"
#include "test/test_reduce.h"
#include "test/test_tsne.h"
#include "test/test_math.h"


namespace {

    // Pairwise Distance Tests
    // TEST(PairwiseDist, 4x2) {test_pairwise_distance(4,2);}
    // TEST(PairwiseDist, 64x2) {test_pairwise_distance(64,2);}
    // TEST(PairwiseDist, 64x64) {test_pairwise_distance(64,64);}
    // TEST(PairwiseDist, 64x128) {test_pairwise_distance(64,128);}
    // TEST(PairwiseDist, 20000x128) {test_pairwise_distance_speed(20000,128);}

    // Reduction Tests
    // TEST(Reductions, ReduceSum_Col_512x512){test_reduce_sum_col(512,512);}
    // TEST(Reductions, ReduceSum_Row_512x512){test_reduce_sum_row(512,512);}
    // TEST(Reductions, ReduceMean_Col_512x512){test_reduce_mean_col(512,512);}
    // TEST(Reductions, ReduceMean_Row_512x512){test_reduce_mean_row(512,512);}
    // TEST(Reductions, ReduceAlpha_Col_512x512_pos){test_reduce_alpha_col(512,512,0.1);}
    // TEST(Reductions, ReduceAlpha_Row_512x512_pos){test_reduce_alpha_row(512,512,0.1);}
    // TEST(Reductions, ReduceAlpha_Col_512x512_neg){test_reduce_alpha_col(512,512,-0.1);}
    // TEST(Reductions, ReduceAlpha_Row_512x512_neg){test_reduce_alpha_row(512,512, -0.1);}

    // TEST(ComputePIJ, CPU_256x50) {test_cpu_compute_pij(256, 50);}
    // TEST(ComputePIJ, GPUisCPU) {test_cpu_is_gpu_pij(16,16);}
    // TEST(ComputePerplexity, GPUisCPU) {test_cpu_is_gpu_perplexity(16,16);}
    // TEST(ComputeSigma, GPUisCPU) {test_cpu_sigmas_search(16,16);}
    // TEST(ComputePerplexity, 16x16) {test_sigmas_search(16, 16);}

    // T-SNE tests
    // TEST(NaiveTSNE, 256x50) {test_tsne(256, 50);}
    // TEST(NaiveTSNE, 512x50) {test_tsne(512, 50);}
    // TEST(NaiveTSNE, 1024x50) {test_tsne(1024, 50);}

    // Test Symmetrization
    // TEST(MatrixSymmetry, 70000x784) {test_sym_mat(70000,784);}

    // Test the BHTSNE
    // TEST(BhTSNE, friendship) {test_bhtsne(70000, 784);}
    //TEST(BhTSNE, friendship) {test_bhtsne(5000, 50);}

    //Test the BHTSNE on MNIST
    // TEST(BhTSNEMnist, friendship) {test_bhtsne_mnist("../mnist2500x768.txt");}
    TEST(BhTSNEMnist, test_set) {test_bhtsne_full_mnist("../train-images.idx3-ubyte");}

    // Test the BHTSNE on CIFAR
   // TEST(BhTSNECifar, train_set_cifar10) {test_bhtsne_full_cifar10("../cifar-10/bin_data");}
    //TEST(BhTSNECifar, train_set_cifar100) {test_bhtsne_full_cifar100("../cifar-100/train.bin");}
    // Test the BHTSNE_Ref
    //TEST(BhTSNERef, friendship) {test_bhtsne_ref(300, 50);}
    
}
