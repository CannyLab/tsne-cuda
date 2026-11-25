/**
 * @brief Unit Tests for the CUDA-TSNE Library
 * 
 * @file test.cu
 * @author David Chan
 * @date 2018-04-04
 */

#include "common.h"
#include "util/cuda_utils.h"
#include "util/reduce_utils.h"
#include <time.h>

// G-Test include
#include "gtest/gtest.h"

// Testing scripts
#include "test/test_reduce.h"


namespace {

    // Reduction Tests
    TEST(Reductions, ReduceSum_Col_512x512){test_reduce_sum_col(512,512);}
    TEST(Reductions, ReduceSum_Row_512x512){test_reduce_sum_row(512,512);}
    TEST(Reductions, ReduceMean_Col_512x512){test_reduce_mean_col(512,512);}
    TEST(Reductions, ReduceMean_Row_512x512){test_reduce_mean_row(512,512);}
    // Non-square shapes, so dividing by the wrong dimension fails (#144).
    TEST(Reductions, ReduceSum_Col_384x512){test_reduce_sum_col(384,512);}
    TEST(Reductions, ReduceSum_Row_384x512){test_reduce_sum_row(384,512);}
    TEST(Reductions, ReduceMean_Col_384x512){test_reduce_mean_col(384,512);}
    TEST(Reductions, ReduceMean_Row_384x512){test_reduce_mean_row(384,512);}
    TEST(Reductions, ReduceMean_Col_512x384){test_reduce_mean_col(512,384);}
    TEST(Reductions, ReduceMean_Row_512x384){test_reduce_mean_row(512,384);}
    TEST(Reductions, ReduceAlpha_Col_512x512_pos){test_reduce_alpha_col(512,512,0.1);}
    TEST(Reductions, ReduceAlpha_Row_512x512_pos){test_reduce_alpha_row(512,512,0.1);}
    TEST(Reductions, ReduceAlpha_Col_512x512_neg){test_reduce_alpha_col(512,512,-0.1);}
    TEST(Reductions, ReduceAlpha_Row_512x512_neg){test_reduce_alpha_row(512,512, -0.1);}

}
