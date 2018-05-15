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
import os
import pkg_resources

def ord_string(s):
    b = bytearray()
    arr = b.extend(map(ord, s))
    return N.array([x for x in b] + [0]).astype(N.uint8)

class TSNE(object):
    def __init__(self,
                 n_components=2,
                 perplexity=50.0,
                 early_exaggeration=2.0,
                 learning_rate=200.0,
                 num_neighbors=1023,
                 force_magnify_iters=250,
                 pre_momentum=0.5,
                 post_momentum=0.8,
                 theta=0.5,
                 epssq=0.0025,
                 n_iter=1000,
                 n_iter_without_progress=1000,
                 min_grad_norm=1e-7,
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
            ):
        """Initialization method for barnes hut T-SNE class.
        """

        # Initialize the variables
        self.n_components = int(n_components)
        if self.n_components != 2:
            raise ValueError('The current barnes-hut implementation does not support projection into dimensions other than 2 for now.')
        self.perplexity = float(perplexity)
        self.early_exaggeration = float(early_exaggeration)
        self.learning_rate = float(learning_rate)
        self.n_iter = int(n_iter)
        self.n_iter_without_progress = int(n_iter_without_progress)
        self.min_grad_norm = float(min_grad_norm)
        if metric not in ['euclidean']:
            raise ValueError('Non-Euclidean metrics are not currently supported. Please use metric=\'euclidean\' for now.')
        else:
            self.metric = metric
        if init not in ['random']:
            raise ValueError('Non-Random initialization is not currently supported. Please use init=\'random\' for now.')
        else:
            self.init = init
        self.verbose = int(verbose)

        # Initialize non-sklearn variables
        self.num_neighbors = int(num_neighbors)
        self.force_magnify_iters = int(force_magnify_iters)
        self.perplexity_epsilon = float(perplexity_epsilon)
        self.pre_momentum = float(pre_momentum)
        self.post_momentum = float(post_momentum)
        self.theta = float(theta)
        self.epssq =float(epssq)
        self.device = int(device)
        self.print_interval = int(print_interval)

        # Point dumpoing
        self.dump_file = str(dump_file)
        self.dump_points = bool(dump_points)
        self.dump_interval = int(dump_interval)

        # Viz
        self.use_interactive = bool(use_interactive)
        self.viz_server = str(viz_server)
        self.viz_timeout = int(viz_timeout)

        # Return style
        if return_style not in ['once','snapshots']:
            raise ValueError('Invalid return style...')
        elif return_style == 'once':
            self.return_style = 0
        elif return_style == 'snapshots':
            self.return_style = 1
        self.num_snapshots = int(num_snapshots)

        # Build the hooks for the BH T-SNE library
        self._path = pkg_resources.resource_filename('tsnecuda','') # Load from current location
        # self._faiss_lib = N.ctypeslib.load_library('libfaiss', self._path) # Load the ctypes library
        # self._gpufaiss_lib = N.ctypeslib.load_library('libgpufaiss', self._path) # Load the ctypes library
        self._lib = N.ctypeslib.load_library('libtsnecuda', self._path) # Load the ctypes library

        # Hook the BH T-SNE function
        self._lib.pymodule_bh_tsne.restype = None
        self._lib.pymodule_bh_tsne.argtypes = [ 
                N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS, WRITEABLE'), # result
                N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'), # points
                ctypes.POINTER(N.ctypeslib.c_intp), # dims
                ctypes.c_float, # Perplexity
                ctypes.c_float, # Learning Rate
                ctypes.c_float, # Magnitude Factor
                ctypes.c_int, # Num Neighbors
                ctypes.c_int, # Iterations
                ctypes.c_int, # Iterations no progress
                ctypes.c_int, # Force Magnify iterations
                ctypes.c_float, # Perplexity search epsilon
                ctypes.c_float, # pre-exaggeration momentum
                ctypes.c_float, # post-exaggeration momentum
                ctypes.c_float, # Theta
                ctypes.c_float, # epssq
                ctypes.c_float, # Minimum gradient norm
                ctypes.c_int, # Initialization types
                N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS'), # Initialization Data
                ctypes.c_bool, # Dump points
                N.ctypeslib.ndpointer(N.uint8, flags='ALIGNED, CONTIGUOUS'), # Dump File
                ctypes.c_int, # Dump interval
                ctypes.c_bool, # Use interactive
                N.ctypeslib.ndpointer(N.uint8, flags='ALIGNED, CONTIGUOUS'), # Viz Server
                ctypes.c_int, # Viz timeout
                ctypes.c_int, # Verbosity
                ctypes.c_int, # Print interval
                ctypes.c_int, # GPU Device
                ctypes.c_int, # Return style
                ctypes.c_int ] # Number of snapshots

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed output.
        
        Arguments:
            X {array} -- Input array, shape: (n_points, n_dimensions) 
        
        Keyword Arguments:
            y {None} -- Ignored (default: {None})
        """

        # Setup points/embedding requirements
        self.points = N.require(X, N.float32, ['CONTIGUOUS', 'ALIGNED'])
        self.embedding = N.zeros(shape=(X.shape[0],self.n_components))
        self.embedding = N.require(self.embedding , N.float32, ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])

        # Handle Initialization
        if y is None:
            self.initialization_type = 1
            self.init_data = N.require(N.zeros((1,1)),N.float32,['CONTIGUOUS','ALIGNED'])
        else:
            self.initialization_type = 3
            self.init_data = N.require(y, N.float32, ['F_CONTIGUOUS', 'ALIGNED'])

        # Handle dumping and viz strings
        self.dump_file_ = N.require(ord_string(self.dump_file), N.uint8, ['CONTIGUOUS', 'ALIGNED'])
        self.viz_server_ = N.require(ord_string(self.viz_server), N.uint8, ['CONTIGUOUS', 'ALIGNED'])


        self._lib.pymodule_bh_tsne(
                self.embedding, # result
                self.points, # points
                self.points.ctypes.shape, # dims
                ctypes.c_float(self.perplexity), # Perplexity
                ctypes.c_float(self.learning_rate), # Learning Rate
                ctypes.c_float(self.early_exaggeration), # Magnitude Factor
                ctypes.c_int(self.num_neighbors), # Num Neighbors
                ctypes.c_int(self.n_iter), # Iterations
                ctypes.c_int(self.n_iter_without_progress), # Iterations no progress
                ctypes.c_int(self.force_magnify_iters), # Force Magnify iterations
                ctypes.c_float(self.perplexity_epsilon), # Perplexity search epsilon
                ctypes.c_float(self.pre_momentum), # pre-exaggeration momentum
                ctypes.c_float(self.post_momentum), # post-exaggeration momentum
                ctypes.c_float(self.theta), # Theta
                ctypes.c_float(self.epssq), # epssq
                ctypes.c_float(self.min_grad_norm), # Minimum gradient norm
                ctypes.c_int(self.initialization_type), # Initialization types
                self.init_data, # Initialization Data
                ctypes.c_bool(self.dump_points), # Dump points
                self.dump_file_, # Dump File
                ctypes.c_int(self.dump_interval), # Dump interval
                ctypes.c_bool(self.use_interactive), # Use interactive
                self.viz_server_, # Viz Server
                ctypes.c_int(self.viz_timeout), # Viz timeout
                ctypes.c_int(self.verbose), # Verbosity
                ctypes.c_int(self.print_interval), # Print interval
                ctypes.c_int(self.device), # GPU Device
                ctypes.c_int(self.return_style), # Return style
                ctypes.c_int(self.num_snapshots) ) # Number of snapshots

        return self.embedding




        





