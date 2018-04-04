/**
 * @brief Various Thrust transform structs
 * 
 * @file transform_func.h
 * @author David Chan
 * @date 2018-04-04
 */

#ifndef THRUST_TRANFORM_FUNC_H
#define THRUST_TRANFORM_FUNC_H

    #include "common.h"

    namespace tfunc {
        struct square {
            __host__ __device__ double operator()(const float &x) const;
        };

        struct sqrt {
            __host__ __device__ double operator()(const float &x) const;
        };

        struct exp {
            __host__ __device__ double operator()(const float &x) const;
        };

        struct exp_no_zero {
            __host__ __device__ double operator()(const float &x) const;
        };

        struct inv {
            __host__ __device__ double operator()(const float &x) const;
        };

        struct inc {
            __host__ __device__ double operator()(const float &x) const;
        };

        struct inc_inv {
            __host__ __device__ double operator()(const float &x) const;
        };

        struct inc_inv_ignore_zero {
            __host__ __device__ double operator()(const float &x) const;
        };

        struct ln {
            __host__ __device__ double operator()(const float &x) const;
        };

        struct kl {
            __host__ __device__ double operator()(const float &x, const float &y) const;
        };

        struct abs {
            __host__ __device__ double operator()(const float &x) const;
        };
    };
#endif