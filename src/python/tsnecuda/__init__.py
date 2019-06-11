from .TSNE import TSNE


def test():
    import numpy as np
    X = np.random.random((5000, 50))
    TSNE(verbose=1).fit_transform(X)
