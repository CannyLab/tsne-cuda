local C = require 'tsne.ffi'

local function tsne(X, n_components, perplexity, n_iter, angle, n_jobs)

  assert(X:dim() == 2, 'X should be 2D array.')
  assert(X:type() == 'torch.DoubleTensor', 'Only DoubleTensors are supported for now.')
  
  n_components = n_components or 2
  perplexity = perplexity or 30
  n_iter = n_iter or 1000
  n_jobs = n_jobs or 1
  angle = angle or 0.5

  assert(n_components == 2, 'n_components should be 2.')
  
  local N, D = X:size(1), X:size(2)
  local Y = X.new(N, n_components):zero()
   
  C.tsne_run_double(torch.data(X), N, D, torch.data(Y), n_components, perplexity, angle, n_jobs, n_iter)
     
  return Y
end

return tsne 
