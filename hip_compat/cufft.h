// HIP build only: maps the cuFFT names used by tsnecuda to hipFFT.
#ifndef TSNECUDA_HIP_COMPAT_CUFFT_H
#define TSNECUDA_HIP_COMPAT_CUFFT_H

#include <hipfft/hipfft.h>

#define cufftHandle hipfftHandle
#define cufftResult hipfftResult
#define cufftReal hipfftReal
#define cufftComplex hipfftComplex
#define cufftCreate hipfftCreate
#define cufftDestroy hipfftDestroy
#define cufftMakePlan2d hipfftMakePlan2d
#define cufftMakePlanMany hipfftMakePlanMany
#define cufftExecR2C hipfftExecR2C
#define cufftExecC2R hipfftExecC2R

#define CUFFT_R2C HIPFFT_R2C
#define CUFFT_C2R HIPFFT_C2R

#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_INVALID_PLAN HIPFFT_INVALID_PLAN
#define CUFFT_ALLOC_FAILED HIPFFT_ALLOC_FAILED
#define CUFFT_INVALID_TYPE HIPFFT_INVALID_TYPE
#define CUFFT_INVALID_VALUE HIPFFT_INVALID_VALUE
#define CUFFT_INTERNAL_ERROR HIPFFT_INTERNAL_ERROR
#define CUFFT_EXEC_FAILED HIPFFT_EXEC_FAILED
#define CUFFT_SETUP_FAILED HIPFFT_SETUP_FAILED
#define CUFFT_INVALID_SIZE HIPFFT_INVALID_SIZE
#define CUFFT_UNALIGNED_DATA HIPFFT_UNALIGNED_DATA
#define CUFFT_INCOMPLETE_PARAMETER_LIST HIPFFT_INCOMPLETE_PARAMETER_LIST
#define CUFFT_INVALID_DEVICE HIPFFT_INVALID_DEVICE
#define CUFFT_PARSE_ERROR HIPFFT_PARSE_ERROR
#define CUFFT_NO_WORKSPACE HIPFFT_NO_WORKSPACE
#define CUFFT_NOT_IMPLEMENTED HIPFFT_NOT_IMPLEMENTED
#define CUFFT_NOT_SUPPORTED HIPFFT_NOT_SUPPORTED
// Current hipFFT no longer defines its licence result (it was 15, as in cuFFT).
#define CUFFT_LICENSE_ERROR static_cast<hipfftResult>(15)

#endif
