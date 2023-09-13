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

#ifndef ATOMIC_H
#define ATOMIC_H

#include <sycl/sycl.hpp>

namespace tsnecuda
{
/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr, Float version.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <sycl::access::address_space addressSpace = sycl::access::address_space::global_space>
inline float atomic_fetch_add(
    float *addr,
    float operand,
    sycl::memory_order memoryOrder = sycl::memory_order::relaxed)
{
    static_assert(sizeof(float) == sizeof(int), "Mismatched type size");

    // cast address addr (a float pointer) to int pointer
    sycl::atomic<int, addressSpace> obj((sycl::multi_ptr<int, addressSpace>(reinterpret_cast<int *>(addr))));

    int   old__int__value;
    float old_float_value;

    do {
    old__int__value             = obj.load(memoryOrder);                                // load value to int   variable
    old_float_value             = *reinterpret_cast<const float *>(&old__int__value);   // load value to float variable
    const float new_float_value = old_float_value + operand;                            // add operatnd to float variable
    const int   new__int__value = *reinterpret_cast<const int *>(&new_float_value);     // cast address of new_float_value to int pointer
    if (obj.compare_exchange_strong(old__int__value, new__int__value, memoryOrder))
        break;
    } while (true);

    return old_float_value;
}

} // namespace tsnecuda

#endif
