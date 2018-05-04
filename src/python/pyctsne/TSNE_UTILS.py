# Python library headers for the T-SNE library
# Woo!

# Do necessary imports to work with Numpy arrayes
import numpy as N
import ctypes
import os
import pkg_resources

_path = pkg_resources.resource_filename('pyctsne','') # Load from current location
_lib = N.ctypeslib.load_library('libpyctsne', _path) # Load the ctypes library


# Distance function hook
_lib.pymodule_e_dist.restype = None
_lib.pymodule_e_dist.argtypes = [ N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS'),
                                  N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS, WRITEABLE'),
                                  ctypes.POINTER(N.ctypeslib.c_intp)
                                ]
def e_pw_dist(points):
    points = N.require(points, N.float32, ['F_CONTIGUOUS', 'ALIGNED'])
    distances = N.empty(shape=(points.shape[0],points.shape[0]))
    distances = N.require(distances, N.float32, ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
    _lib.pymodule_e_dist(points, distances, points.ctypes.shape)
    return distances

# TSNE Function Hook
_lib.pymodule_naive_tsne.restype = None
_lib.pymodule_naive_tsne.argtypes = [ N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS'), # Input Points
                                  N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS, WRITEABLE'), # Output points
                                  ctypes.POINTER(N.ctypeslib.c_intp), # Input points dimension
                                  ctypes.c_int, # Projected Dimension
                                  ctypes.c_float, # Learning Rate
                                  ctypes.c_float # Perplexity
                                ]
def c_naive_tsne(points, proj_dim, learning_rate, perplexity):
    points = N.require(points, N.float32, ['F_CONTIGUOUS', 'ALIGNED'])
    results = N.empty(shape=(points.shape[0],proj_dim))
    results = N.require(results, N.float32, ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
    _lib.pymodule_naive_tsne(points, results, points.ctypes.shape, c_int(proj_dim), c_float(learning_rate), c_float(perplexity))
    return results


# NAIVE COMPUTE PIJ Hook
_lib.pymodule_compute_pij.restype = None
_lib.pymodule_compute_pij.argtypes = [ N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS'), # Input Points
                                    N.ctypeslib.ndpointer(N.float32, ndim=1, flags='ALIGNED, F_CONTIGUOUS'), # Sigmas
                                    N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS, WRITEABLE'), # PIJ output
                                    ctypes.POINTER(N.ctypeslib.c_intp) # Input points dimension
                                    ]
def c_compute_pij(points, sigmas):
    points = N.require(points, N.float32, ['F_CONTIGUOUS', 'ALIGNED'])
    sigmas = N.require(sigmas, N.float32, ['F_CONTIGUOUS', 'ALIGNED'])
    results = N.empty(shape=(points.shape[0],points.shape[0]))
    results = N.require(results, N.float32, ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
    _lib.pymodule_compute_pij(points, sigmas, results,  points.ctypes.shape)
    return results

# Hook the BH T-SNE snapshot function
_lib.pymodule_bhsnapshot.restype = None
_lib.pymodule_bhsnapshot.argtypes = [ N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, CONTIGUOUS'), # Input Points
                            N.ctypeslib.ndpointer(N.float32, flags='ALIGNED, F_CONTIGUOUS, WRITEABLE'), # Output points
                            ctypes.POINTER(N.ctypeslib.c_intp), # Input points dimension
                            ctypes.c_int, # Projected Dimension
                            ctypes.c_float, # Perplexity
                            ctypes.c_float, # Early exagg
                            ctypes.c_float, # Learning Rate
                            ctypes.c_int, # n_iter
                            ctypes.c_int, # n_iter w/o progress
                            ctypes.c_float, # min_norm
                            N.ctypeslib.ndpointer(N.float32, ndim=2, flags='ALIGNED, F_CONTIGUOUS, WRITEABLE'), # Initialization data
                            ctypes.c_int, # Num snapshots
                        ]        

def tsne_snapshots(points, n_components=2, perplexity=32.0, early_exaggeration=12, learning_rate=500, iterations=1000, y=None, num_snapshots=5):
        if y is None:
            Y = N.random.rand(points.shape[0],2)
        points = N.require(points, N.float32, ['CONTIGUOUS', 'ALIGNED'])
        embedding_ = N.zeros(shape=(points.shape[0],2,num_snapshots))
        embedding_ = N.require(embedding_ , N.float32, ['F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
        Y = N.require(Y, N.float32, ['F_CONTIGUOUS', 'ALIGNED'])
        _lib.pymodule_bhsnapshot(points, embedding_, points.ctypes.shape, 
                                        ctypes.c_int(n_components), 
                                        ctypes.c_float(perplexity), 
                                        ctypes.c_float(early_exaggeration),
                                        ctypes.c_float(learning_rate), 
                                        ctypes.c_int(iterations),
                                        ctypes.c_int(iterations),
                                        ctypes.c_float(0),
                                        Y,
                                        ctypes.c_int(num_snapshots)                                        
                                        )
        return embedding_

    

