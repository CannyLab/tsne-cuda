#!/bin/bash
# Build a tsne-cuda Python wheel inside a CUDA build container.
#   $1 = CUDA local-version tag (e.g. cu129)
#   $2 = base version           (e.g. 4.0.0)
# The wheel is written to /wheels. Wheels are py3-none-any: libtsnecuda.so is
# dlopened via ctypes, so a single wheel works across all Python 3.x for a given
# CUDA build; the +cuXXX local version records which CUDA runtime it needs.
set -ex

CUDA_TAG="${1:?usage: build_and_deploy.sh <cuda_tag> <version>}"
TSNE_VERSION="${2:-4.0.0}"

# Configure + compile the shared library and stage the importable package.
# The internal per-toolkit gencode ladder covers A100/H100/H200/B200; pass
# -DCMAKE_CUDA_ARCHITECTURES to override.
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

# The build stages the package (with libtsnecuda.so) under ./python.
cd python
echo "${TSNE_VERSION}+${CUDA_TAG}" > VERSION.txt

# Build the wheel with PEP 517 build isolation (pip fetches setuptools/wheel into
# a throwaway env). This avoids installing into the base image's Python, which
# newer distros (e.g. Ubuntu 24.04, PEP 668) mark externally-managed.
python3 -m pip wheel . --no-deps -w /wheels
ls -l /wheels
