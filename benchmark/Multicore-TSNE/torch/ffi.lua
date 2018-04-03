local ffi = require 'ffi'

local libpath = package.searchpath('libtsne_multicore', package.cpath)
if not libpath then return end

defs = [[

void tsne_run_double(double* X, int N, int D, 
                     double* Y, int no_dims, 
                     double perplexity, double theta, int num_threads, int max_iter);

]]


ffi.cdef(defs)

return ffi.load(libpath)
