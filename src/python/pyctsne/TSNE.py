# Python library headers for the T-SNE library
# Woo!

# Do necessary imports to work with Numpy arrayes
import numpy as N
import ctypes
import os

_path = os.path.dirname('__file__') # Load from current location
_lib = N.ctypeslib.load_library('libpyctsne', _path) # Load the ctypes library

_lib.pymodule_e_dist.restype = None
_lib.pymodule_e_dist.argtypes = [ N.ctypeslib.ndpointer(N.float32, ndim=2, flags='aligned, contiguous'),
                                  N.ctypeslib.ndpointer(N.float32, ndim=2, flags='aligned, contiguous, writeable'),
                                  ctypes.POINTER(N.ctypeslib.c_intp)
                                ]

def e_pw_dist(points):
    points = N.require(N.transpose(points), N.float32, ['CONTIGUOUS', 'ALIGNED'])
    distances = N.empty(shape=(points.shape[0],points.shape[0]))
    distances = N.require(distances, N.float32, ['CONTIGUOUS', 'ALIGNED', 'WRITEABLE'])
    _lib.pymodule_e_dist(points, distances, points.ctypes.shape)
    return distances
    

