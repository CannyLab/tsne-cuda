# TSNE-CUDA

This repo is an optimized CUDA version of [FIt-SNE algorithm](https://github.com/KlugerLab/FIt-SNE) with associated python modules. We find that our implementation of t-SNE can be up to 1200x faster than Sklearn, or up to 50x faster than Multicore-TSNE when used with the right GPU. The paper describing our approach, as well as the results below, is available at [https://arxiv.org/abs/1807.11824](https://arxiv.org/abs/1807.11824).

You can install binaries with anaconda for CUDA versions 9.0, 9.2, 10.0, and 10.1 using `conda install cuda<major><minor> tsnecuda -c cannylab`. For more details or to install from source, check out our wiki: [https://github.com/CannyLab/tsne-cuda/wiki/](https://github.com/CannyLab/tsne-cuda/wiki/)

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

#### Note: There appear to be some compilation instability issues when using parallel compilation. Running Make twice seems to fix it, as does running make without parallel compile. We believe this should be fixed in the most recent release.

### Run

Like many of the libraries available, the python wrappers subscribe to the same API as [sklearn.manifold.TSNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

You can run it as follows:

```
from tsnecuda import TSNE
X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
```

We only support `n_components=2`. We currently have no plans to support more dimensions as this requires significant changes to the code to accomodate.

For more information on running the library, or using it as a C++ library, see the [Python usage](https://github.com/CannyLab/tsne-cuda/wiki/Basic-Usage:-Python) or [C++ Usage](https://github.com/CannyLab/tsne-cuda/wiki/Basic-Usage:-Cxx) sections of the wiki.

# Citation

Please cite the corresponding paper if it was useful for your research:

```
@article{chan2019gpu,
  title={GPU accelerated t-distributed stochastic neighbor embedding},
  author={Chan, David M and Rao, Roshan and Huang, Forrest and Canny, John F},
  journal={Journal of Parallel and Distributed Computing},
  volume={131},
  pages={1--13},
  year={2019},
  publisher={Elsevier}
}
```

This library is built on top of the following technology, without this tech, none of this would be possible!

[L. Van der Maaten's paper](http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

[FIt-SNE](https://github.com/KlugerLab/FIt-SNE)

[Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)

[BHTSNE](https://github.com/lvdmaaten/bhtsne/)

[CUDA Utilities/Pairwise Distance](https://github.com/OrangeOwlSolutions)

[LONESTAR-GPU](http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu)

[FAISS](https://github.com/facebookresearch/faiss)

[GTest](https://github.com/google/googletest)

[CXXopts](https://github.com/jarro2783/cxxopts)


# License

Our code is built using components from FAISS, the Lonestar GPU library, GTest, CXXopts, and OrangeOwl's CUDA utilities. Each portion of the code is governed by their respective licenses - however our code is governed by the BSD-3 license found in LICENSE.txt
