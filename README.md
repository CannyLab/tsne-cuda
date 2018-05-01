# CUDA TSNE

This repo is a roof-line optimized CUDA version of [Barnes-Hut t-SNE](https://github.com/lvdmaaten/bhtsne) by L. Van der Maaten with associated python modules. 

# Benchmark

TODO :)

# How to use

Todo :)

## Python
### Install

#### Requirements

- CUDA: You will need a working version of the CUDA toolkit which can be obtained from [here](https://developer.nvidia.com/cuda-toolkit).
- CMAKE: Version >= 3.5.1 which can be obtained by running `sudo apt install cmake` on ubuntu/debian systems.
- MKL/OpenBLAS: If you're using MKL, install it using the intel provided installation scripts. If you're using OpenBLAS install it using `sudo apt install libopenblas-dev` on ubuntu/debian systems. 
- GCC/llvm-clang: This is likely already installed on your system. If not, on ubuntu you can run `sudo apt install build-essential` to get a version.
- OpenMP: On ubuntu this is likely already installed with your version of GCC. For other distributions, be sure your compiler has OpenMP support.
- Python (for Python bindings): Python is not required, however to build the python bindings you must install Python. This library was tested with Python 3, however it is possible that Python 2 will work as well (though it is untested). 
- Doxygen: To build the documentation, a working version of doxygen is required (which can be obtained using `sudo apt install doxygen` on debian/ubuntu systems).
- ZMQ: Necessary for building the interactive visualization. TODO: Add CMake flags to turn this off.

This package contains no distribution specific code - thus it should compile and run on other Linux distros (and possibly even OSX) however do so at your own peril, as it has not been tested.

#### Building

To build the C++ portion of the library:
```
git clone https://github.com/rmrao/tsne-cuda.git
cd tsne-cuda/build/
cmake ..
make
```

If you are using OpenBLAS instead of MKL, you need to run the following instead (otherwise it will complain that MKL is missing):
```
git clone https://github.com/rmrao/tsne-cuda.git
cd tsne-cuda/build/
cmake .. -DWITH_MKL=OFF
make
```

For faster builds, you can utilize parallel versions of make by running `make -j<num cores>`, so to build with 5 cores, you would run `make -j5`. This can significantly speed up the build, however sometimes the build will fail because of parallel dependencies. To fix a failed build, just run `make` again, and it should work.


To install the python package, do the above steps, then run:
```
cd python/
python setup.py install
```

To build the documentation, run from the root directory:
```
cd tsne-cuda/build/
cmake .. -DBUILD_DOC=ON
make doc_doxygen
```
This command will put the documentation in the `tsne-cuda/build/doc_doxygen/html` folder. 

#### Known good build configurations

- Ubuntu 16.04, Python 3.5.2, GCC 5.4.0, CUDA 8.0.61, Intel MKL 18.0.1

### Run

Like many of the libraries available, the python wrappers subscribe to the same API as [sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

You can run it as follows:

```
from pyctsne import TSNE

Embedded_Data = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
```

It's worth noting that if n_components is > 3, then the program uses the naive O(n^2) method by default. To try to run the experimental FMM based method, you can use TSNE(method='experimental').

### Test

The library can be tested using ctest (bundled with CMAKE) by running the following

```
cd ${ROOT}/build/
ctest
```

# License

TODO: Update this portion of the code

Inherited from [original repo's license](https://github.com/lvdmaaten/bhtsne).

# Future work

- Allow for double precision
- Expand FMM methods
- Add multi-threaded CPU version for those without a GPU

# Known Bugs

- Tests seg-fault when run with not enough stack space. Use 'ulimit -s unlimited' as a temporary workaround.

# Citation

Please cite this repository if it was useful for your research:

```
@misc{cudatsne2018,
  author = {Chan, D. and Rao, R. and Huang, Z.},
  title = {TSNE-CUDA},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rmrao/tsne-cuda.git}},
}
```

This library is built on top of the following technology, without this tech, none of this would be possible!

[L. Van der Maaten's paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

[Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)

[BHTSNE](https://github.com/lvdmaaten/bhtsne/)

[CUDA Utilities/Pairwise Distance](https://github.com/OrangeOwlSolutions)

[FAISS](https://github.com/facebookresearch/faiss)

[GTest](https://github.com/google/googletest)
