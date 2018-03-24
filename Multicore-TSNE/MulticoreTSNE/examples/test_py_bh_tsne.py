# install tsne form this repo first
# https://github.com/danielfrg/tsne
from tsne import bh_sne
import gzip
import pickle
import numpy as np
import matplotlib
from cycler import cycler
import urllib
import os
import sys

import time

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_mnist():

    if not os.path.exists('mnist.pkl.gz'):
        print('downloading MNIST')
        urllib.urlretrieve(
            'http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz')
        print('downloaded')

    f = gzip.open("mnist.pkl.gz", "rb")
    if sys.version_info >= (3, 0):
        train, val, test = pickle.load(f, encoding='latin1')
    else:
        train, val, test = pickle.load(f)
    f.close()

    # Get all data in one array
    _train = np.asarray(train[0], dtype=np.float64)
    _val = np.asarray(val[0], dtype=np.float64)
    _test = np.asarray(test[0], dtype=np.float64)
    mnist = np.vstack((_train, _val, _test))

    # Also the classes, for labels in the plot later
    classes = np.hstack((train[1], val[1], test[1]))

    return mnist, classes


def plot(Y, classes, name):
    digits = set(classes)
    fig = plt.figure()
    colormap = plt.cm.spectral
    plt.gca().set_prop_cycle(
        cycler('color', [colormap(i) for i in np.linspace(0, 0.9, 10)]))
    ax = fig.add_subplot(111)
    labels = []
    for d in digits:
        idx = classes == d
        ax.plot(Y[idx, 0], Y[idx, 1], 'o')
        labels.append(d)
    ax.legend(labels, numpoints=1, fancybox=True)
    fig.savefig(name)

################################################################


mnist, classes = get_mnist()

mnist_tsne = bh_sne(mnist)

plot(mnist_tsne, classes, 'tsne_pybhtsne_core.png')
