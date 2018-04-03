===========
PyCTSNE
===========

PyCTSNE provides a roofline optimized CUDA implementation of the T-SNE algorithm presented in [CITE]. PyCTSNE is able to compute the T-SNE of large numbers of points up to XX times faster than other leading libraries, and provides simple python bindings with a SKLearn style interface::

    #!/usr/bin/env python

    from PyCTSNE import TSNE
    embeddedX = TSNE(n_components=2).fit_transform(X)
