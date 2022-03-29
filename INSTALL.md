# Installation

## Conda installation

Conda binaries are available through conda-forge! To install tsne-cuda with conda, run `conda install tsnecuda -c conda-forge`.

## Pip installation

Pip binaries are available for Python >=3.6 and CUDA 10.1, 10.2, 11.0, 11.1, 11.2 and 11.3 and FAISS version 1.6.5 and Intel MKL 2018.

To install tsne-cuda with pip, run `pip3 install tsnecuda`. This installs tsnecuda with a CUDA version of 10.2. For this to work, you MUST have both FAISS version 1.6.5
and intel MKL installed already on your machine - see installation instructions for FAISS: [here](https://github.com/facebookresearch/faiss/blob/v1.6.5/INSTALL.md).
and installation instructions for MKL on Ubuntu/debian [here](https://github.com/eddelbuettel/mkl4deb)

When installing FAISS from source, it is necessary to build and install both the C++ and the python requirements.

If you do not want to use 10.2, you can download using pip from our hosted sources:
```
# CUDA 11.0
pip3 install tsnecuda==3.0.1+cu110 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
# CUDA 11.1
pip3 install tsnecuda==3.0.1+cu111 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
# CUDA 11.2
pip3 install tsnecuda==3.0.1+cu112 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
# CUDA 11.3
pip3 install tsnecuda==3.0.1+cu113 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
# CUDA 10.1
pip3 install tsnecuda==3.0.1+cu101 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
# CUDA 10.0
pip3 install tsnecuda==3.0.1+cu100 -f https://tsnecuda.isx.ai/tsnecuda_stable.html
```


## From source

A number of requirements are necessary for building our code from source.

- CUDA: You will need a working version of the CUDA toolkit which can be obtained from here. Our code has been tested compiling with CUDA versions 9.0, 9.1, 9.2, 10.0, 10.1, 10.2, 11.0 and 11.2. Other versions - may not be supported.
- CMAKE: Version >= 3.20.0 which can be obtained by running sudo apt install cmake on ubuntu/debian systems.
- MKL/OpenBLAS: If you're using MKL, install it using the intel provided installation scripts. If you're using OpenBLAS install it using sudo apt install libopenblas-dev on ubuntu/debian systems.
- GCC/llvm-clang: This is likely already installed on your system. If not, on ubuntu you can run sudo apt install build-essential to get a version.
- OpenMP: On ubuntu this is likely already installed with your version of GCC. For other distributions, be sure your compiler has OpenMP support.
- Python (for Python bindings) >= 3.6: Python is not required, however to build the python bindings you must install Python.
- Gflags >= 2.2: On ubuntu, you can install with `sudo apt install lilbgflags-dev`
- Gtest >= 1.10: On ubuntu, you can install with `sudo apt install libgtest-dev`
- FAISS >= 1.6.5: This can be installed by following the instructions (https://github.com/facebookresearch/faiss/blob/v1.6.5/INSTALL.md)[here]

Optional:
  - Doxygen: To build the documentation, a working version of doxygen is required (which can be obtained using sudo apt install doxygen on debian/ubuntu systems).
  - ZMQ: Necessary for building the interactive visualization. On ubuntu you can obtain ZMQ by using sudo apt install libzmq-dev.


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
- -WITH_MKL: (DEFAULT OFF) Build with MKL support. If your MKL is installed in the default location "/opt/intel/mkl" then this is the only argument you need to change. If MKL is installed somewhere else, you must also pass the root MKL directory with -DMKL_DIR=<root directory> . If this is off, you must have OpenBLAS installed.
- -DCMAKE_CXX_COMPILER,-DCMAKE_C_COMPILER (DEFAULT system default) It is possible on newer systems that you will get a compatability error "NVCC does not support GCC versions greater than 6.4.0". To fix this error, you can install an older compiler, and use these cmake options to build the library.

Broken features (will cause cmake/compilation issues):
- -DBUILD_TEST: (DEFAULT OFF) Build the test suite. To turn this on, use -DBUILD_TEST=TRUE
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

### Installing the python bindings

Once you have compiled the python bindings you can install the Python bindings by changing into the `build/python` directory, and running:
```
pip install -e .
```
### Validating the Install

Unfortunately, the current set of tests do not compile. The best way to determine that everything is working is to run

```python
import tsnecuda
tsnecuda.test()
```
This does a t-SNE on 5000 points, so it should complete relatively quickly (1-2 seconds). If there are no error messages and it doesn't hang, you should be good to go.
