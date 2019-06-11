/**
 * @brief Utility/Error Checking code.
 *
 * @file cuda_utils.cu
 * @date 2018-04-04
 * Copyright (C) 2012-2017 Orange Owl Solutions.
 */

 /*
 CUDA Utilities - Utilities for high performance CPUs/C/C++ and GPUs/CUDA computing library.

    Copyright (C) 2012-2017 Orange Owl Solutions.


    CUDA Utilities is free software: you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDA Utilities is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Bluebird Library.  If not, see <http://www.gnu.org/licenses/>.


    For any request, question or bug reporting please visit http://www.orangeowlsolutions.com/
    or send an e-mail to: info@orangeowlsolutions.com
*/


#ifndef CUDA_UTILITIES_CUH
#define CUDA_UTILITIES_CUH

#include "common.h"

__host__ __device__ int iDivUp(int, int);
extern "C" void CublasSafeCall(cublasStatus_t);
extern "C" void CusparseSafeCall(cusparseStatus_t err);
extern "C" void CufftSafeCall(cufftResult err);
extern "C" void GpuErrorCheck(cudaError_t ans);

#endif
