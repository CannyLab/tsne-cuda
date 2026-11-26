# Installation

## Conda installation

Conda binaries for Linux and Windows (x86_64) are available through conda-forge. To install tsne-cuda with conda, run `conda install tsnecuda -c conda-forge`.

## Pip installation

Prebuilt wheels for Linux (x86_64) are attached to the [4.1.0 release](https://github.com/CannyLab/tsne-cuda/releases/tag/4.1.0) and listed at https://tsnecuda.isx.ai/tsnecuda_stable.html. 4.1.1 changed only the Windows build, so it has no wheels of its own. One wheel works with every Python 3 version. Wheels don't include FAISS or the CUDA libraries, so install those first. Install the FAISS version the wheel was built against: FAISS's C++ ABI changes between releases, and a wheel won't load against a different version.

| Wheel | Native code for | Runtime libraries from conda-forge |
|---|---|---|
| `4.1.0+cu118` | sm_52 to sm_86 (Maxwell to Ampere) | `"libfaiss=1.9.0=*cuda118*"` |
| `4.1.0+cu124` | sm_70 to sm_90 (Volta to Hopper) | `"libfaiss=1.9.0=*cuda126*" libcufft libcusparse` |
| `4.1.0+cu126` | sm_70 to sm_90 (Volta to Hopper) | `"libfaiss=1.9.0=*cuda126*" libcufft libcusparse` |
| `4.1.0+cu129` | sm_70 to sm_120 (Volta to Blackwell) | `"libfaiss=1.9.0=*cuda126*" libcufft libcusparse` |
| `4.1.0+cu130` | sm_75 to sm_120 (Turing to Blackwell) | `"libfaiss=1.14.3=*cuda130*" libcufft libcusparse` |

Each wheel also carries PTX (compute_80 in cu118, compute_90 in cu124 and cu126, compute_120 in cu129 and cu130), so GPUs newer than the list compile the kernels on first use. Maxwell and Pascal GPUs need the cu118 wheel or a source build.

For example, with the CUDA 12.9 wheel:
```
conda create -n tsnecuda -c conda-forge python=3.12 "libfaiss=1.9.0=*cuda126*" libcufft libcusparse
conda activate tsnecuda
pip install https://github.com/CannyLab/tsne-cuda/releases/download/4.1.0/tsnecuda-4.1.0+cu129-py3-none-any.whl
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda activate tsnecuda
```
The library looks for FAISS and the CUDA libraries on `LD_LIBRARY_PATH`. The `conda env config vars` line sets it each time the environment is activated. Without it, using `TSNE` fails with an error naming `libtsnecuda.so`.

`conda install -c conda-forge tsnecuda` installs all of this for you, for CUDA 12.9 and 13.x on Linux and Windows.


## From source

A number of requirements are necessary for building our code from source.

- CUDA: You will need a working version of the CUDA toolkit which can be obtained from here. Our code has been tested compiling with CUDA versions 11.x, 12.x, and 13.x. The build targets modern datacenter GPUs out of the box (A100, H100/H200, B200) and honors an externally supplied `-DCMAKE_CUDA_ARCHITECTURES=...`. Other versions may not be supported.
- CMAKE: Version >= 3.20.0 which can be obtained by running sudo apt install cmake on ubuntu/debian systems.
- MKL/OpenBLAS: If you're using MKL, install it using the intel provided installation scripts. If you're using OpenBLAS install it using sudo apt install libopenblas-dev on ubuntu/debian systems.
- GCC/llvm-clang: This is likely already installed on your system. If not, on ubuntu you can run sudo apt install build-essential to get a version.
- OpenMP: On ubuntu this is likely already installed with your version of GCC. For other distributions, be sure your compiler has OpenMP support.
- Python (for Python bindings) >= 3.10: Python is not required, however to build the python bindings you must install Python. The python module is the supported entry point.
- FAISS (GPU) >= 1.7: This can be installed from conda-forge with `conda install -c conda-forge libfaiss` (recommended), or built from source per the FAISS instructions.

Optional:
  - Doxygen: To build the documentation, a working version of doxygen is required (which can be obtained using sudo apt install doxygen on debian/ubuntu systems).

The standalone C++ CLI and the ZMQ-based interactive visualization are no longer built (the `gflags`, `gtest`, `cxxopts`, and `zmq` dependencies have been dropped) to keep the build lean and portable; use the Python module instead.


First, clone the repository, and change into the cloned directory using:
```
git clone https://github.com/rmrao/tsne-cuda.git && cd tsne-cuda
```

Next, initialize the submodules from the root directory using:
```
git submodule init
git submodule update
```

Next, change in the build directory:
```
cd build/
```

From the build directory, we can configure our project. There are a number of options that may be necessary:
- -DBUILD_PYTHON: (DEFAULT ON) Build the python package. This is necessary for the python bindings.
- -DBUILD_TESTS: (DEFAULT OFF) Build the GPU unit tests (`tsnecuda_test`, run with `ctest`). GoogleTest is used from the system if found, otherwise downloaded at configure time.
- -WITH_MKL: (DEFAULT OFF) Build with MKL support. If your MKL is installed in the default location "/opt/intel/mkl" then this is the only argument you need to change. If MKL is installed somewhere else, you must also pass the root MKL directory with -DMKL_DIR=<root directory> . If this is off, you must have OpenBLAS installed.
- -DCMAKE_CXX_COMPILER,-DCMAKE_C_COMPILER (DEFAULT system default) It is possible on newer systems that you will get a compatability error "NVCC does not support GCC versions greater than 6.4.0". To fix this error, you can install an older compiler, and use these cmake options to build the library.

Broken features (will cause cmake/compilation issues):
- -DWITH_ZMQ: (DEFAULT OFF) There is a bug when using GCC version >= 6.0 with nvcc which means that ZMQ cannot be properly compiled. Thus, to avoid this bug which is present in Ubuntu versions 17.10 and 18.04 by default, you must use -DWITH_ZMQ=FALSE. CUDA 10.0 recently fixed this, so you can use this feature with CUDA 10.0 if you have installed ZMQ correctly.

To configure, use the following CMAKE command:
```
cmake .. CMAKE_ARGS
```
where the CMAKE_ARGS are taken from the above. You almost certainly just want to do `cmake ..` with no arguments.

Finally, to build the library use:
```
make
```
For speedy compilation (using multiple threads), you can use
```
make -j<num cores>
```
Using multiple threads may throw errors in the compilation due to nonexistent files. To fix this, just run a single threaded `make` after compilation completes.

### Building on Windows

On Windows, nvcc uses MSVC (`cl.exe`) as its host compiler, so you need Visual Studio 2022 or the Build Tools for Visual Studio 2022 with the "Desktop development with C++" workload. The simplest way to get FAISS and the CUDA libraries is from conda-forge. Run these commands in an "x64 Native Tools Command Prompt for VS 2022" so that `cl.exe` is on `PATH`:
```
conda create -n tsnecuda-build -c conda-forge python=3.12 cmake ninja "libfaiss=*=*_cuda" cuda-nvcc cuda-cudart-dev libcublas-dev libcufft-dev libcusparse-dev
conda activate tsnecuda-build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH="%CONDA_PREFIX%\Library" ^
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
pip install .\build\python
```
Replace `89` with your GPU's compute capability without the dot (for example 86 for an RTX 3090), or leave the option out to build for every architecture the toolkit supports, which takes much longer. Blackwell GPUs (100, 120) need CUDA 13 on Windows: CUDA 12's headers don't compile for them with MSVC. A CUDA 12 build still runs on Blackwell, because the default list includes sm_90 PTX, which the driver compiles on first use. The build produces `build\python\tsnecuda\libtsnecuda.dll`. Before it loads the DLL, `TSNE` adds the conda environment's `Library\bin` and the CUDA toolkit's `bin` directory (from `CUDA_PATH`) to the DLL search path, so `faiss.dll` and the CUDA DLLs load without changes to `PATH`.

### Building for AMD GPUs (ROCm)

tsne-cuda can also be built with HIP for AMD GPUs. In addition to the requirements above (without the CUDA toolkit), this needs:

- ROCm with hipBLAS, hipFFT, hipSPARSE and rocThrust.
- FAISS built with GPU support for ROCm (`-DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_ROCM=ON`) and installed, see the FAISS install instructions.

Configure with `-DUSE_HIP=ON`, the target GPU architecture, and the ROCm and FAISS install prefixes, then build:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_PREFIX_PATH="/opt/rocm;<faiss install prefix>"
cmake --build build -j
```
Replace `gfx942` with the architecture of your GPU (`rocm_agent_enumerator` lists it); several architectures can be given separated by semicolons. If CMake does not find the HIP compiler, add `-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`, or put `/opt/rocm/bin` on `PATH`. The Python package is then in `build/python` as for the CUDA build. A FAISS built with the ROCm clang links LLVM's OpenMP runtime `libomp.so`, which the loader may not find when tsne-cuda itself is built with GCC; if using `tsnecuda` fails with `OSError: libomp.so: cannot open shared object file`, add `/opt/rocm/llvm/lib` to `LD_LIBRARY_PATH`.

#### Windows

The HIP build also works on Windows with the ROCm SDK. Use the GNU-style `clang`/`clang++` from the ROCm SDK for the C, C++ and HIP compilers (not `clang-cl`), Ninja, and a FAISS built for ROCm on Windows:
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 ^
  -DCMAKE_C_COMPILER=<rocm>/lib/llvm/bin/clang.exe ^
  -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang++.exe ^
  -DCMAKE_HIP_COMPILER=<rocm>/lib/llvm/bin/clang++.exe ^
  -DCMAKE_PREFIX_PATH="<rocm>;<faiss install prefix>"
cmake --build build
```
This produces `libtsnecuda.dll`. Python does not search `PATH` for the DLLs it depends on, so make `faiss.dll` and the ROCm runtime DLLs loadable, for example by copying them next to `libtsnecuda.dll` or with `os.add_dll_directory()` before importing `tsnecuda`.

### Installing the python bindings

Once you have compiled the python bindings you can install the Python bindings by changing into the `build/python` directory, and running:
```
pip install -e .
```
### Validating the Install

The quickest way to determine that everything is working is to run

```python
import tsnecuda
tsnecuda.test()
```
This does a t-SNE on 5000 points, so it should complete relatively quickly (1-2 seconds). If there are no error messages and it doesn't hang, you should be good to go.

For a more thorough check of embedding quality on the GPU, install `pytest` and `scikit-learn` and run the Python tests:
```
python -m pytest -v build/python/tsnecuda/test
```
The large clustering test embeds 50000 points; set `TSNECUDA_TEST_BLOBS_N` to a smaller number on GPUs with little memory.
