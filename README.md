# TSNE-CUDA

***WARNING: This code is still in active development. While the core code is tested and working, some features need aditional testing.***

This repo is an optimized CUDA version of [Barnes-Hut t-SNE](https://github.com/lvdmaaten/bhtsne) by L. Van der Maaten with associated python modules. We find that our implementation of t-SNE can be up to 1200x faster than Sklearn, or up to 50x faster than Multicore-TSNE when used with the right GPU. The paper describing our approach, as well as the results below, is available at [https://arxiv.org/abs/1807.11824](https://arxiv.org/abs/1807.11824).

To begin, check out our wiki for install instructions and usage: [https://github.com/CannyLab/tsne-cuda/wiki/](https://github.com/CannyLab/tsne-cuda/wiki/)

# Benchmarks
### Simulated Data
![](docs/simulated_speedup.png)

Time taken compared to other state of the art algorithms on synthetic datasets with 50 dimensions and four clusters for varying numbers of points. Note the log scale on both the points and time axis, and that the scale of the x-axis is in thousands of points (thus, the values on the x-axis range from 1K to 10M points. Dashed lines represent projected times. Projected scaling assumes an O(nlog(n)) implementation.

### MNIST
![](docs/mnist_speedup.png)

The performance of t-SNE-CUDA compared to other state-of-the-art implementations on the MNIST dataset. t-SNE-CUDA runs on the raw pixels of the MNIST dataset (60000 images x 768 dimensions) in under 7 seconds.

### CIFAR
![](docs/cifar_speedup.png)

The performance of t-SNE-CUDA compared to other state-of-the-art implementations on the CIFAR-10 dataset. t-SNE-CUDA runs on the raw pixels of the CIFAR-10 training set (50000 images x 1024 dimensions x 3 channels) in under 12 seconds.

### Comparison of Embedding Quality
The quality of the embeddings produced by t-SNE-CUDA do not differ significantly from the state of the art implementations. See below for a comparison of MNIST cluster outputs.

![](docs/mnist_comparison.jpg)

Left: MULTICORE-4 (501s), Middle: BH-TSNE (1156s), Right: t-SNE-CUDA (Ours, 6.98s).

# Installation

To install our library, follow the instructions in the [installation section](https://github.com/CannyLab/tsne-cuda/wiki/Installation) of the wiki.

### Run

Like many of the libraries available, the python wrappers subscribe to the same API as [sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

You can run it as follows:

```
from tsnecuda import TSNE
X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
```

It's worth noting that if n_components is >= 3, then the program uses the naive O(n^2) method by default. If the number of components is 2, then you can use the heavily optimized Barnes-Hut implementation.

For more information on running the library, or using it as a C++ library, see the [Python usage](https://github.com/CannyLab/tsne-cuda/wiki/Basic-Usage:-Python) or [C++ Usage](https://github.com/CannyLab/tsne-cuda/wiki/Basic-Usage:-Cxx) sections of the wiki.

# Future work

- Allow for double precision
- Expand FMM methods
- Add multi-threaded CPU version for those without a GPU

# Known Bugs

- Odd bug with some datasets that causes a hang/gpu memory error. 

# Citation

Please cite this repository if it was useful for your research:

```
@article{chan2018t,
  title={t-SNE-CUDA: GPU-Accelerated t-SNE and its Applications to Modern Data},
  author={Chan, David M and Rao, Roshan and Huang, Forrest and Canny, John F},
  journal={arXiv preprint arXiv:1807.11824},
  year={2018}
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

Our code is built using components from FAISS, the Lonestar GPU library, GTest, CXXopts, and OrangeOwl's CUDA utilities. Each portion of the code is governed by their respective licenses - however our code is governed by the BSD-3 license found in LICENSE.txt
