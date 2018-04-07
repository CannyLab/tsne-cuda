# CUDA TSNE

This repo is a roof-line optimized CUDA version of [Barnes-Hut t-SNE](https://github.com/lvdmaaten/bhtsne) by L. Van der Maaten with associated python modules. 

# Benchmark

TODO :)

# How to use

Todo :)

## Python
### Install

Make sure `cmake` is installed on your system, and you will also need a C++ compiler, such as `gcc` or `llvm-clang`. On macOS, you can get both via [homebrew](https://brew.sh/). In addition, you will need NVCC and a suitable GPU. 

To build the C++ portion of the library:
```
git clone https://github.com/rmrao/tsne-cuda.git
cd tsne-cuda/build/
cmake ..
make
```

To install the python package, do the above steps, then run:
```
cd python/
python setup.py install
```

This is tested working on Ubuntu 16.04 with Python 3. 

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

# Citation

Please cite this repository if it was useful for your research:

```
@misc{Ulyanov2016,
  author = {Chan, D. and Rao, R. and Huang, Z.},
  title = {TSNE-CUDA},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rmrao/tsne-cuda.git}},
}
```

The following work was extremely helpful in building this library: 

[L. Van der Maaten's paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

[Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)

[BHTSNE](https://github.com/lvdmaaten/bhtsne/)

[CUDA Utilities/Pairwise Distance](https://github.com/OrangeOwlSolutions)
