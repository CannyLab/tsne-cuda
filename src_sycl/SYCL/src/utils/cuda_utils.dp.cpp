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

#include <sycl/sycl.hpp>
#include "include/utils/cuda_utils.h"

/*******************/
/* iDivUp FUNCTION */
/*******************/
//extern "C" int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

// /********************/
// /* CUDA ERROR CHECK */
// /********************/
// // --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
// void gpuAssert(int code, const char *file, int line, bool abort = true)
// {
// }

// extern "C" void GpuErrorCheck(int ans) { gpuAssert((ans), __FILE__, __LINE__); }

// /*************************/
// /* CUBLAS ERROR CHECKING */
// /*************************/
// static const char *_cublasGetErrorEnum(int error)
// {
// 	switch (error)
// 	{
//         case 0:
//                 return "CUBLAS_STATUS_SUCCESS";

//         case 1:
//                 return "CUBLAS_STATUS_NOT_INITIALIZED";

//         case 3:
//                 return "CUBLAS_STATUS_ALLOC_FAILED";

//         case 7:
//                 return "CUBLAS_STATUS_INVALID_VALUE";

//         case 8:
//                 return "CUBLAS_STATUS_ARCH_MISMATCH";

//         case 11:
//                 return "CUBLAS_STATUS_MAPPING_ERROR";

//         case 13:
//                 return "CUBLAS_STATUS_EXECUTION_FAILED";

//         case 14:
//                 return "CUBLAS_STATUS_INTERNAL_ERROR";

//         case 15:
//                 return "CUBLAS_STATUS_NOT_SUPPORTED";

//         case 16:
//                 return "CUBLAS_STATUS_LICENSE_ERROR";
// 	}

// 	return "<unknown>";
// }

// inline void __CublasSafeCall(int err, const char *file, const int line)
// {
//         if (0 != err) {
//                 fprintf(stderr, "CUBLAS error in file '%s', line %d, error: %s\nterminating!\n", __FILE__, __LINE__, \
// 			_cublasGetErrorEnum(err)); \
// 			assert(0); \
// 	}
// }

// extern "C" void CublasSafeCall(int err) {
//  __CublasSafeCall(err, __FILE__, __LINE__);
// }

// /***************************/
// /* CUSPARSE ERROR CHECKING */
// /***************************/
// static const char *_cusparseGetErrorEnum(int error)
// {
// 	switch (error)
// 	{

//         case 0:
//                 return "CUSPARSE_STATUS_SUCCESS";

//         case 1:
//                 return "CUSPARSE_STATUS_NOT_INITIALIZED";

//         case 2:
//                 return "CUSPARSE_STATUS_ALLOC_FAILED";

//         case 3:
//                 return "CUSPARSE_STATUS_INVALID_VALUE";

//         case 4:
//                 return "CUSPARSE_STATUS_ARCH_MISMATCH";

//         case 5:
//                 return "CUSPARSE_STATUS_MAPPING_ERROR";

//         case 6:
//                 return "CUSPARSE_STATUS_EXECUTION_FAILED";

//         case 7:
//                 return "CUSPARSE_STATUS_INTERNAL_ERROR";

//         case 8:
//                 return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

//         case 9:
//                 return "CUSPARSE_STATUS_ZERO_PIVOT";
// 	}

// 	return "<unknown>";
// }

// inline void __CusparseSafeCall(int err, const char *file, const int line)
// {
//         if (0 != err) {
//                 fprintf(stderr, "CUSPARSE error in file '%s', line %d, error %s\nterminating!\n", __FILE__, __LINE__, \
// 			_cusparseGetErrorEnum(err)); \
// 			assert(0); \
// 	}
// }

// extern "C" void CusparseSafeCall(int err) {
//  __CusparseSafeCall(err, __FILE__, __LINE__);
// }

// /************************/
// /* CUFFT ERROR CHECKING */
// /************************/

// static const char *_cufftGetErrorEnum(int error)
// {
//     switch (error)
//     {
//         case 0:
//             return "CUFFT_SUCCESS";

//         case 1:
//             return "CUFFT_INVALID_PLAN";

//         case 2:
//             return "CUFFT_ALLOC_FAILED";

//         case 3:
//             return "CUFFT_INVALID_TYPE";

//         case 4:
//             return "CUFFT_INVALID_VALUE";

//         case 5:
//             return "CUFFT_INTERNAL_ERROR";

//         case 6:
//             return "CUFFT_EXEC_FAILED";

//         case 7:
//             return "CUFFT_SETUP_FAILED";

//         case 8:
//             return "CUFFT_INVALID_SIZE";

//         case 9:
//             return "CUFFT_UNALIGNED_DATA";

//         case 11:
//             return "CUFFT_INVALID_DEVICE";

//         case 10:
//             return "CUFFT_INCOMPLETE_PARAMETER_LIST";

//         case 12:
//             return "CUFFT_PARSE_ERROR";

//         case 13:
//             return "CUFFT_NO_WORKSPACE";

//         case 16:
//             return "CUFFT_NOT_SUPPORTED";

//         case 14:
//             return "CUFFT_NOT_IMPLEMENTED";

//         case 15:
//             return "CUFFT_LICENSE_ERROR";

//         default:
//             return "<unknown>";
//     }
// }

// inline void __cufftSafeCall(int err, const char *file, const int line)
// {
//     if (0 != err) {
//                 fprintf(stderr,
//                 "CUFFT error in file '%s', line %d, error %s\nterminating!\n",
//                 __FILE__,
//                 __LINE__,
// 				_cufftGetErrorEnum(err));
//                 dpct::get_current_device().reset();
//         assert(0);
//     }
// }

// extern "C" void CufftSafeCall(int err) {
//  __cufftSafeCall(err, __FILE__, __LINE__);
// }

// /// END OF CUDA UTILITIES