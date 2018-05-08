"""Bindings for the naive O(n^2) TSNE algorithm

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

class NaiveTSNE(object):
    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0, 
                    learning_rate=1.0, n_iter=1000, n_iter_without_progress=300,
                    min_grad_norm=1e-7, metric='euclidean',init='random',verbose=0,
                    random_seed=None):
        """Initialization method for naive T-SNE class.
        
        Keyword Arguments:
            n_components {int} -- Dimension of the embedded space. (default: {2})
            perplexity {int} -- The perplexity is related to the number of nearest neighbors that is
                                    used in other manifold learning algorithms. Larger datasets usually 
                                    require a larger perplexity. Consider selecting a value between 5 and 
                                    50. The choice is not extremely critical since t-SNE is quite insensitive 
                                    to this parameter. (default: {30})
            early_exaggeration {float} -- Controls how tight natural clusters in the original space are in
                                                the embedded space and how much space will be between them. 
                                                For larger values, the space between natural clusters will 
                                                be larger in the embedded space. Again, the choice of this 
                                                parameter is not very critical. If the cost function increases 
                                                during initial optimization, the early exaggeration factor or
                                                the learning rate might be too high. (default: {12.0})
            learning_rate {float} -- The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the
                                        learning rate is too high, the data may look like a ‘ball’ with any point
                                        approximately equidistant from its nearest neighbours. If the learning 
                                        rate is too low, most points may look compressed in a dense cloud with 
                                        few outliers. If the cost function gets stuck in a bad local minimum 
                                        increasing the learning rate may help. (default: {1.0})
            n_iter {int} -- Maximum number of iterations for the optimization. Should be at least 250. (default: {1000})
            n_iter_without_progress {int} -- Maximum number of iterations without progress before we abort the 
                                                optimization, used after 250 initial iterations with early 
                                                exaggeration. Note that progress is only checked every 50 
                                                iterations so this value is rounded to the next multiple of 50. (default: {300})
            min_grad_norm {float} -- If the gradient norm is below this threshold, the optimization will be stopped. (default: {1e-7})
            metric {str} -- TODO: Implement this feature. Currently only the euclidean metric (L2) is supported. (default: {'euclidean'})
            init {str} -- TODO: Initialization of embedding. Possible options are ‘random’, ‘pca’, and a numpy array of shape 
                                (n_samples, n_components). PCA initialization cannot be used with precomputed distances
                                 and is usually more globally stable than random initialization. (default: {'random'})
            verbose {int} -- TODO: Verbosity level. (default: {0})
            random_seed {int} -- TODO: [The seed used by the random number generator. (default: {None})
        """

        # Initialize the variables
        self.n_components = int(n_components)
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
            self.metric = metric
        self.verbose = int(verbose)
        if random_seed is not None:
            self.random_seed = int(random_seed)
        else:
            self.random_seed = os.urandom()

        # Build the hooks for the Naive T-SNE library
        self._path = pkg_resources.resource_filename('tsnecuda','') # Load from current location
        self._lib = N.ctypeslib.load_library('libtsnecuda', _path) # Load the ctypes library

        # Hook the naive T-SNE function
        self._lib.pymodule_naive_tsne.restype = None
        self._lib.pymodule_naive_tsne.argtypes = [ N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS'), # Input Points
                                  N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS, WRITEABLE'), # Output points
                                  ctypes.POINTER(N.ctypeslib.c_intp), # Input points dimension
                                  ctypes.c_int, # Projected Dimension
                                  ctypes.c_float, # Perplexity
                                  ctypes.c_float, # Early exagg
                                  ctypes.c_float, # Learning Rate
                                  ctypes.c_int, # n_iter
                                  ctypes.c_int, # n_iter w/o progress
                                  ctypres.c_float # min_norm
                                ]

        # Set up the attributed
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = None

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed output.
        
        Arguments:
            X {array} -- Input array, shape: (n_points, n_dimensions) 
        
        Keyword Arguments:
            y {None} -- Ignored (default: {None})
        """

        X = N.require(X, N.float32, ['F_CONTIGUOUS', 'ALIGNED'])
        self.embedding_ = N.empty(shape=(X.shape[0],self.n_components))
        self.embedding_ = N.require(results, N.float32, ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
        self._lib.pymodule_naive_tsne(X, self.embedding_, X.ctypes.shape, 
                                        c_int(self.n_components), 
                                        c_float(self.perplexity), 
                                        c_float(self.early_exaggeration),
                                        c_float(self.learning_rate), 
                                        c_int(self.n_iter),
                                        c_int(self.n_iter_without_progress),
                                        c_float(self.min_grad_norm))
        return self.embedding_




        





