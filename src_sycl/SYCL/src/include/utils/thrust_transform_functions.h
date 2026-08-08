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

#include <sycl/sycl.hpp>
/**
 * @brief Various transformations for thrust::transform
 * 
 * @file thrust_transform_functions.h
 * @author David Chan
 * @date 2018-05-05
 * Copyright (c) 2018, Regents of the University of California
 */

#ifndef SRC_INCLUDE_UTIL_THRUST_TRANSFORM_FUNCTIONS_H_
#define SRC_INCLUDE_UTIL_THRUST_TRANSFORM_FUNCTIONS_H_

namespace tsnecuda {
namespace utils {

struct FunctionalEntropy {
  float operator()(const float& x) const {
      float val = x * sycl::log((float)x);
      return (x == 0 || val != val || sycl::isinf(val) || sycl::isinf(val)) ? 0 : val;
    }
};

struct FunctionalSquare {
    float operator()(const float& x) const {
        return x * x;
    }
};

struct FunctionalAbs {
    float operator()(const float& x) const {
        return sycl::fabs((float)x);
    }
};

}  // namespace utils
}  // namespace tsnecuda

#endif  // SRC_INCLUDE_UTIL_THRUST_TRANSFORM_FUNCTIONS_H_
