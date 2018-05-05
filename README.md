# TSNE-CUDA

This repo is an optimized CUDA version of [Barnes-Hut t-SNE](https://github.com/lvdmaaten/bhtsne) by L. Van der Maaten with associated python modules. We find that our implementation of t-SNE can be up to 1200x faster than Sklearn, or up to 50x faster than Multicore-TSNE when used with the right GPU.

# Benchmarks

TODO :)

# Installation

To install our library, follow the instructions in the [installation section](https://github.com/rmrao/tsne-cuda/wiki/Installation) of the wiki.

### Run

Like many of the libraries available, the python wrappers subscribe to the same API as [sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

You can run it as follows:

```
from pyctsne import TSNE
X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
```

It's worth noting that if n_components is >= 3, then the program uses the naive O(n^2) method by default. If the number of components is 2, then you can use the heavily optimized Barnes-Hut implementation.

For more information on running the library, or using it as a C++ library, see the [Python usage](https://github.com/rmrao/tsne-cuda/wiki/Basic-Usage:-Python) or [C++ Usage](https://github.com/rmrao/tsne-cuda/wiki/Basic-Usage:-Cxx) sections of the wiki.

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

[LONESTAR-GPU](http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu)

[FAISS](https://github.com/facebookresearch/faiss)

[GTest](https://github.com/google/googletest)

[CXXopts](https://github.com/jarro2783/cxxopts)


# License

Our code is built using components from FAISS, the Lonestar GPU library, GTest, CXXopts, and OrangeOwl's CUDA utilities. Each portion of the code is governed by their respective licenses - however our code is governed by the MIT license found in LICENSE.txt
