===========
tsnecuda
===========

tsnecuda provides an optimized CUDA implementation of the T-SNE algorithm by L Van der Maaten. tsnecuda is able to compute the T-SNE of large numbers of points up to 1200 times faster than other leading libraries, and provides simple python bindings with a SKLearn style interface::

    #!/usr/bin/env python

    from tsnecuda import TSNE
    embeddedX = TSNE(n_components=2).fit_transform(X)

For more information, check out the repository at https://github.com/rmrao/tsne-cuda. 
