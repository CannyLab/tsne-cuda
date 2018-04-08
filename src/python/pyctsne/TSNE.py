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
    

