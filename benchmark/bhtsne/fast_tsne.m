function mappedX = fast_tsne(X, no_dims, initial_dims, perplexity, theta, alg, max_iter)
%FAST_TSNE Runs the C++ implementation of Barnes-Hut t-SNE
%
%   mappedX = fast_tsne(X, no_dims, initial_dims, perplexity, theta, alg)
%
% Runs the C++ implementation of Barnes-Hut-SNE. The high-dimensional 
% datapoints are specified in the NxD matrix X. The dimensionality of the 
% datapoints is reduced to initial_dims dimensions using PCA (default = 50)
% before t-SNE is performed. Next, t-SNE reduces the points to no_dims
% dimensions. The perplexity of the input similarities may be specified
% through the perplexity variable (default = 30). The variable theta sets
% the trade-off parameter between speed and accuracy: theta = 0 corresponds
% to standard, slow t-SNE, while theta = 1 makes very crude approximations.
% Appropriate values for theta are between 0.1 and 0.7 (default = 0.5).
% The variable alg determines the algorithm used for PCA. The default is set 
% to 'svd'. Other options are 'eig' or 'als' (see 'doc pca' for more details).
% The function returns the two-dimensional data points in mappedX.
%
% NOTE: The function is designed to run on large (N > 5000) data sets. It
% may give poor performance on very small data sets (it is better to use a
% standard t-SNE implementation on such data).


% Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 1. Redistributions of source code must retain the above copyright
%    notice, this list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the
%    documentation and/or other materials provided with the distribution.
% 3. All advertising materials mentioning features or use of this software
%    must display the following acknowledgement:
%    This product includes software developed by the Delft University of Technology.
% 4. Neither the name of the Delft University of Technology nor the names of 
%    its contributors may be used to endorse or promote products derived from 
%    this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
% OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
% OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
% EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
% BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
% IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
% OF SUCH DAMAGE.


    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
    if ~exist('initial_dims', 'var') || isempty(initial_dims)
        initial_dims = 50;
    end
    if ~exist('perplexity', 'var') || isempty(perplexity)
        perplexity = 30;
    end
    if ~exist('theta', 'var') || isempty(theta)
        theta = 0.5;
    end
    if ~exist('alg', 'var') || isempty(alg)
        alg = 'svd';
    end
    if ~exist('max_iter', 'var') || isempty(max_iter)
       max_iter=1000; 
    end
    
    % Perform the initial dimensionality reduction using PCA
    X = double(X);
    X = bsxfun(@minus, X, mean(X, 1));
    M = pca(X,'NumComponents',initial_dims,'Algorithm',alg);
    X = X * M;
    
    tsne_path = which('fast_tsne');
    tsne_path = fileparts(tsne_path);
    
    % Compile t-SNE C code
    if(~exist(fullfile(tsne_path,'./bh_tsne'),'file') && isunix)
        system(sprintf('g++ %s %s -o %s -O2',...
            fullfile(tsne_path,'./sptree.cpp'),...
            fullfile(tsne_path,'./tsne.cpp'),...
            fullfile(tsne_path,'./bh_tsne')));
    end

    % Run the fast diffusion SNE implementation
    write_data(X, no_dims, theta, perplexity, max_iter);
    tic
    [flag, cmdout] = system(['"' fullfile(tsne_path,'./bh_tsne') '"']);
    if(flag~=0)
        error(cmdout);
    end
    toc
    [mappedX, landmarks, costs] = read_data;   
    landmarks = landmarks + 1;              % correct for Matlab indexing
    delete('data.dat');
    delete('result.dat');
end


% Writes the datafile for the fast t-SNE implementation
function write_data(X, no_dims, theta, perplexity, max_iter)
    [n, d] = size(X);
    h = fopen('data.dat', 'wb');
	fwrite(h, n, 'integer*4');
	fwrite(h, d, 'integer*4');
    fwrite(h, theta, 'double');
    fwrite(h, perplexity, 'double');
	fwrite(h, no_dims, 'integer*4');
    fwrite(h, max_iter, 'integer*4');
    fwrite(h, X', 'double');
	fclose(h);
end


% Reads the result file from the fast t-SNE implementation
function [X, landmarks, costs] = read_data
    h = fopen('result.dat', 'rb');
	n = fread(h, 1, 'integer*4');
	d = fread(h, 1, 'integer*4');
	X = fread(h, n * d, 'double');
    landmarks = fread(h, n, 'integer*4');
    costs = fread(h, n, 'double');      % this vector contains only zeros
    X = reshape(X, [d n])';
	fclose(h);
end
