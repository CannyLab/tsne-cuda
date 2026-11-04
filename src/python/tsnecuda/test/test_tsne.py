"""GPU tests of embedding quality.

t-SNE is not bitwise reproducible (the kernels accumulate with atomics), so
these tests check quality metrics rather than exact values. They need a GPU
and scikit-learn. Run them with

    python -m pytest -v build/python/tsnecuda/test

TSNECUDA_TEST_BLOBS_N sets the size of the large clustering test (default
50000); lower it on GPUs with little memory.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.datasets import load_digits, make_blobs  # noqa: E402
from sklearn.manifold import TSNE as SklearnTSNE, trustworthiness  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from tsnecuda import TSNE  # noqa: E402

TRUSTWORTHINESS_MIN = 0.97
KNN_ACCURACY_MIN = 0.95
SKLEARN_TRUSTWORTHINESS_GAP = 0.02


def _digits():
    X, y = load_digits(return_X_y=True)
    return X.astype(np.float32), y


def _embed_digits():
    X, _ = _digits()
    return TSNE(perplexity=30, learning_rate=200, n_iter=1000,
                random_seed=0).fit_transform(X)


def _digits_metrics(Y):
    X, y = _digits()
    knn = KNeighborsClassifier(n_neighbors=10)
    return {
        "finite": bool(np.isfinite(Y).all()),
        "trustworthiness": float(trustworthiness(X, Y, n_neighbors=10)),
        "knn_accuracy": float(cross_val_score(knn, Y, y, cv=5).mean()),
    }


def _check_digits(metrics):
    assert metrics["finite"]
    assert metrics["trustworthiness"] >= TRUSTWORTHINESS_MIN, metrics
    assert metrics["knn_accuracy"] >= KNN_ACCURACY_MIN, metrics


def _blobs_ari(n_samples, num_neighbors=32):
    X, labels = make_blobs(n_samples=n_samples, centers=10, n_features=50,
                           cluster_std=2.0, random_state=0)
    Y = TSNE(num_neighbors=num_neighbors, random_seed=0).fit_transform(
        X.astype(np.float32))
    assert np.isfinite(Y).all()
    # KMeans can crash inside OpenBLAS when OpenMP starts a thread per core on
    # hosts with very many cores; a few threads are plenty for 2-D points.
    with threadpool_limits(limits=8):
        predicted = KMeans(n_clusters=10, n_init=10,
                           random_state=0).fit_predict(Y)
    return adjusted_rand_score(labels, predicted)


def _digits_in_subprocess(env_overrides):
    # Kernel synchronization is read once per process, so environment
    # variants run in a fresh interpreter. pytest may have put the package on
    # sys.path itself, so pass its parent directory to the child explicitly.
    env = dict(os.environ, **env_overrides)
    package_parent = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [package_parent, env.get("PYTHONPATH")]))
    code = ("import json; from tsnecuda.test import test_tsne as t; "
            "print(json.dumps(t._digits_metrics(t._embed_digits())))")
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_random_points_finite():
    X = np.random.RandomState(0).random_sample((5000, 50))
    Y = TSNE().fit_transform(X)
    assert Y.shape == (5000, 2)
    assert np.isfinite(Y).all()


def test_digits_quality():
    _check_digits(_digits_metrics(_embed_digits()))


def test_digits_matches_sklearn():
    X, _ = _digits()
    ours = trustworthiness(X, _embed_digits(), n_neighbors=10)
    reference = trustworthiness(
        X, SklearnTSNE(perplexity=30, init="random", random_state=0)
        .fit_transform(X), n_neighbors=10)
    assert ours >= reference - SKLEARN_TRUSTWORTHINESS_GAP, (ours, reference)


def test_digits_quality_synchronized_kernels():
    _check_digits(_digits_in_subprocess({"TSNE_SYNC_KERNELS": "1"}))


def test_digits_quality_launch_overrides():
    _check_digits(_digits_in_subprocess({
        "TSNE_FFT_THREADS": "64",
        "TSNE_ATTR_THREADS": "256",
        "TSNE_REP_THREADS": "256",
        "TSNE_INTEG_THREADS": "256",
    }))


def test_blobs_clusters():
    n_samples = int(os.environ.get("TSNECUDA_TEST_BLOBS_N", "50000"))
    assert _blobs_ari(n_samples) >= 0.95


def test_blobs_clusters_cpu_neighbors():
    # 1024 or more neighbors takes the CPU FAISS search path. With fewer
    # points the probed cells can hold fewer than 1024 points, and FAISS then
    # pads the neighbor list with -1.
    assert _blobs_ari(10000, num_neighbors=1024) >= 0.9
