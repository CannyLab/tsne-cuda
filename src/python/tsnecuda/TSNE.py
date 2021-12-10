"""Bindings for the Barnes Hut TSNE algorithm with fast nearest neighbors

Refs:
References
[1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
[2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
http://homepage.tudelft.nl/19j49/t-SNE.html

"""

import numpy as N
import ctypes
from ctypes import c_int, c_float, c_bool, POINTER
import os
from collections import namedtuple


def ord_string(s):
    b = bytearray()
    arr = b.extend(map(ord, s))
    return N.array([x for x in b] + [0]).astype(N.uint8)


TsneConfig = namedtuple(
    'TsneConfig',
    ['result', 'points', 'dims', 'perplexity', 'learning_rate', 'early_exaggeration',
     'magnitude_factor', 'num_neighbors', 'iterations', 'iterations_no_progress',
     'force_magnify_iters', 'perplexity_search_epsilon', 'pre_exaggeration_momentum',
     'post_exaggeration_momentum', 'theta', 'epssq', 'min_gradient_norm', 'initialization_type',
     'preinit_data', 'dump_points', 'dump_file', 'dump_interval', 'use_interactive',
     'viz_server', 'viz_timeout', 'verbosity', 'print_interval', 'gpu_device', 'return_style',
     'num_snapshots', 'distance_metric'])


class TSNE(object):

    def __init__(self,
                 n_components=2,
                 perplexity=50.0,
                 early_exaggeration=12.0,
                 learning_rate=200.0,
                 num_neighbors=32,
                 force_magnify_iters=250,
                 pre_momentum=0.5,
                 post_momentum=0.8,
                 theta=0.5,
                 epssq=0.0025,
                 n_iter=1000,
                 n_iter_without_progress=1000,
                 min_grad_norm=0.0,
                 perplexity_epsilon=1e-3,
                 metric='euclidean',
                 init='random',
                 return_style='once',
                 num_snapshots=5,
                 verbose=0,
                 random_seed=None,
                 use_interactive=False,
                 viz_timeout=10000,
                 viz_server="tcp://localhost:5556",
                 dump_points=False,
                 dump_file="dump.txt",
                 dump_interval=1,
                 print_interval=10,
                 device=0,
                 magnitude_factor=5):
        """Initialization method for barnes hut T-SNE class.
        """

        # Initialize the variables
        self.n_components = int(n_components)
        if self.n_components != 2:
            raise NotImplementedError('The current fit-tsne implementation does not support '
                                      'projection into dimensions other than 2 for now.')
        if metric in ('euclidean', 'L2'):
            self.metric = 1
        elif metric == 'innerproduct':
            self.metric = 0
        # elif metric == 'L1':
        #     self.metric = 3
        # elif metric == 'LInf':
        #     self.metric = 4
        # elif metric == 'canberra':
        #     self.metric = 20
        # elif metric == 'braycurtis':
        #     self.metric = 5
        # elif metric == 'jensenshannon':
        #     self.metric = 6
        else:
            raise NotImplementedError(
                "Only 'euclidean' and 'innerproduct' metrics are currently supported.")
        if init not in ['random']:
            raise NotImplementedError("Non-Random initialization is not currently supported. "
                                      "Please use init='random' for now.")

        self.perplexity = float(perplexity)
        self.learning_rate = float(learning_rate)
        self.early_exaggeration = float(early_exaggeration)
        self.iterations = int(n_iter)
        self.iterations_no_progress = int(n_iter_without_progress)
        self.min_gradient_norm = float(min_grad_norm)
        self.init = init
        self.verbosity = int(verbose)

        # Initialize non-sklearn variables
        self.num_neighbors = int(num_neighbors)
        self.force_magnify_iters = int(force_magnify_iters)
        self.perplexity_search_epsilon = float(perplexity_epsilon)
        self.pre_exaggeration_momentum = float(pre_momentum)
        self.post_exaggeration_momentum = float(post_momentum)
        self.theta = float(theta)
        self.epssq = float(epssq)
        self.gpu_device = int(device)
        self.print_interval = int(print_interval)
        self.magnitude_factor = float(magnitude_factor)

        # Point dumping
        self.dump_file = str(dump_file)
        self.dump_points = bool(dump_points)
        self.dump_interval = int(dump_interval)

        # Viz
        self.use_interactive = bool(use_interactive)
        self.viz_server = str(viz_server)
        self.viz_timeout = int(viz_timeout)

        # Return style
        if return_style not in ['once', 'snapshots']:
            raise ValueError('Invalid return style...')
        elif return_style == 'once':
            self.return_style = 0
        elif return_style == 'snapshots':
            self.return_style = 1
        self.num_snapshots = int(num_snapshots)

        # Build the hooks for the BH T-SNE library
        self._path = os.path.dirname(__file__)
        self._lib = N.ctypeslib.load_library(
            'libtsnecuda', self._path)  # Load the ctypes library

        # Hook the BH T-SNE function
        self._lib.pymodule_tsne.restype = None
        tsne_argtypes = TsneConfig(
            result=N.ctypeslib.ndpointer(
                N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS, WRITEABLE'),
            points=N.ctypeslib.ndpointer(
                N.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'),
            dims=POINTER(N.ctypeslib.c_intp),
            perplexity=c_float,
            learning_rate=c_float,
            early_exaggeration=c_float,
            magnitude_factor=c_float,
            num_neighbors=c_int,
            iterations=c_int,
            iterations_no_progress=c_int,
            force_magnify_iters=c_int,
            perplexity_search_epsilon=c_float,
            pre_exaggeration_momentum=c_float,
            post_exaggeration_momentum=c_float,
            theta=c_float,
            epssq=c_float,
            min_gradient_norm=c_float,
            initialization_type=c_int,
            preinit_data=N.ctypeslib.ndpointer(
                N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS'),
            dump_points=c_bool,
            dump_file=N.ctypeslib.ndpointer(
                N.uint8, flags='ALIGNED, CONTIGUOUS'),
            dump_interval=c_int,
            use_interactive=c_bool,
            viz_server=N.ctypeslib.ndpointer(
                N.uint8, flags='ALIGNED, CONTIGUOUS'),
            viz_timeout=c_int,
            verbosity=c_int,
            print_interval=c_int,
            gpu_device=c_int,
            return_style=c_int,
            num_snapshots=c_int,
            distance_metric=c_int,
        )

        self._lib.pymodule_tsne.argtypes = list(tsne_argtypes)

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed output.

        Arguments:
            X {array} -- Input array, shape: (n_points, n_dimensions)

        Keyword Arguments:
            y {None} -- The initialization to use for the T-SNE (default: {None})
        """

        # Setup points/embedding requirements
        self.points = N.require(X, N.float32, ['CONTIGUOUS', 'ALIGNED'])
        self.embedding = N.zeros(shape=(X.shape[0], self.n_components))
        self.embedding = N.require(self.embedding, N.float32, [
                                   'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])

        # Handle Initialization
        if y is None:
            self.initialization_type = 1
            self.preinit_data = N.require(N.zeros((1, 1)), N.float32, [
                                          'CONTIGUOUS', 'ALIGNED'])
        else:
            self.initialization_type = 3
            self.preinit_data = N.require(
                y, N.float32, ['F_CONTIGUOUS', 'ALIGNED'])

        # Handle dumping and viz strings
        self.dump_file_ = N.require(ord_string(self.dump_file), N.uint8, [
                                    'CONTIGUOUS', 'ALIGNED'])
        self.viz_server_ = N.require(ord_string(self.viz_server), N.uint8, [
                                     'CONTIGUOUS', 'ALIGNED'])

        tsne_args = self._construct_tsne_args(self.embedding, self.points)

        self._lib.pymodule_tsne(*tsne_args)
        return self.embedding

    def _construct_tsne_args(self, result_array, input_array):
        tsne_args = TsneConfig(
            result=result_array,
            points=input_array,
            dims=input_array.ctypes.shape,
            perplexity=c_float(self.perplexity),
            learning_rate=c_float(self.learning_rate),
            early_exaggeration=c_float(self.early_exaggeration),
            magnitude_factor=c_float(self.magnitude_factor),
            num_neighbors=c_int(self.num_neighbors),
            iterations=c_int(self.iterations),
            iterations_no_progress=c_int(self.iterations_no_progress),
            force_magnify_iters=c_int(self.force_magnify_iters),
            perplexity_search_epsilon=c_float(self.perplexity_search_epsilon),
            pre_exaggeration_momentum=c_float(self.pre_exaggeration_momentum),
            post_exaggeration_momentum=c_float(
                self.post_exaggeration_momentum),
            theta=c_float(self.theta),
            epssq=c_float(self.epssq),
            min_gradient_norm=c_float(self.min_gradient_norm),
            initialization_type=c_int(self.initialization_type),
            preinit_data=self.preinit_data,
            dump_points=c_bool(self.dump_points),
            dump_file=ord_string(self.dump_file),
            dump_interval=c_int(self.dump_interval),
            use_interactive=c_bool(self.use_interactive),
            viz_server=ord_string(self.viz_server),
            viz_timeout=c_int(self.viz_timeout),
            verbosity=c_int(self.verbosity),
            print_interval=c_int(self.print_interval),
            gpu_device=c_int(self.gpu_device),
            return_style=c_int(self.return_style),
            num_snapshots=c_int(self.num_snapshots),
            distance_metric=c_int(self.metric)
        )

        return tsne_args
