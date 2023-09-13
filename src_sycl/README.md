# tsne

tsne implements [FIt-SNE algorithm](https://github.com/KlugerLab/FIt-SNE) for various GPU architectures (original CUDA source code is from [here](https://github.com/CannyLab/tsne-cuda)).

## SYCL version

- The CUDA code was converted to SYCL using Intel's DPC++ Compatiblity Tool (DPCT) available [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html).
- The same SYCL code runs on Intel GPUs & CPUs as well as NVIDIA (tested on A100 and H100) and AMD (tested on MI100 and MI250) GPUs. See build instructions below for more details.
- NOTE #1: This version bypasses use of FAISS by running input images through an offline Python version of FAISS and using its output as input to this SYCL version. So this is more suitable for hardware and framework (SYCL, CUDA, HIP) benchmarking.
- NOTE #2: This version also does not use fft from MKL. Instead it uses a manually implemented fft. For apples-to-apples comparison, we do have a corresponding (modified) CUDA version available [here](https://github.com/oneapi-src/Velocity-Bench/tree/main/tsne) in [Velocity-Bench](https://github.com/oneapi-src/Velocity-Bench). I am happy to add that CUDA version here, if that will be useful.

# Current Version:
- Initial release of the workload

# Build Instructions
Notes
- icpx compiler mentioned below is included in the oneAPI Base Toolkit available [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
- clang++ compiler mentioned below is available [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md).


For Intel GPU -  
First source icpx compiler. Then,

```
cd src_sycl/SYCL
mkdir build
cd build
CXX=icpx cmake -DGPU_AOT=pvc ..
make -sj
```
Note:
- To enable AOT compilation, please use the flag `-DGPU_AOT=pvc` for PVC.

For AMD GPU -  
First source clang++ compiler. Then,
```
cd src_sycl/SYCL
mkdir build
cd build
CXX=clang++ cmake -DUSE_AMDHIP_BACKEND=gfx90a ..
make -sj
```
Note:
- We use the flag `-DUSE_AMDHIP_BACKEND=gfx90a` for MI250. Use the correct value for your GPU.

For NVIDIA GPU -  
First source clang++ compiler. Then,
```
cd src_sycl/SYCL
mkdir build
cd build
CXX=clang++ cmake -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80 ..
make -sj
```
Note:
- We use the flag `-DUSE_SM=80` for A100 or `-DUSE_SM=90` for H100.

# Run instructions

After building, to run the workload, cd into the SYCL/build folder, if not already there. Then

```
# PVC 1 tile:
ONEAPI_DEVICE_SELECTOR=level_zero:0.0 ./tsne
```
```
# PVC 2 tiles:
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./tsne
```
```
# AMD GPU:
ONEAPI_DEVICE_SELECTOR=hip:0 ./tsne
```
```
# NVIDIA GPU:
ONEAPI_DEVICE_SELECTOR=cuda:0 ./tsne
```

# Output

Output gives the total time for running the whole workload.
