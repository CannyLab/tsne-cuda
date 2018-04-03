import numpy as np
import sys
from MulticoreTSNE import MulticoreTSNE as TSNE

for N in [1000, 10000, 100000]:
    for D in [10, 100, 1000, 10000]:

        print ('=====================')
        print ('N: %d, D: %d' % (N, D))
        print ('=====================')
        X = np.random.rand(N, D)
        tsne = TSNE(n_jobs=int(sys.argv[1]))
        mnist_tsne = tsne.fit_transform(X)
