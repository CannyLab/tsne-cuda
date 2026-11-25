// HIP build only: maps the cuBLAS names used by tsnecuda to hipBLAS.
#ifndef TSNECUDA_HIP_COMPAT_CUBLAS_V2_H
#define TSNECUDA_HIP_COMPAT_CUBLAS_V2_H

#include <hipblas/hipblas.h>

#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSgemm hipblasSgemm
#define cublasSgemv hipblasSgemv
#define cublasSsyrk hipblasSsyrk

#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T

#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
// hipBLAS has no licence status; map it to a value no other case uses.
#define CUBLAS_STATUS_LICENSE_ERROR HIPBLAS_STATUS_UNKNOWN

#endif
