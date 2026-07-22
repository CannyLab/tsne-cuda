#!/bin/bash
# Build tsne-cuda Python wheels for the supported CUDA toolkits and collect them
# into ./wheelhouse. Each wheel is py3-none-any and tagged +cuXXX (the CUDA
# runtime it needs); a single wheel serves every Python 3.x for that CUDA build.
#
# tsne-cuda 4.0.0 targets modern datacenter GPUs (A100/H100/H200/B200), so the
# matrix drops the CUDA 10.x / early-11.x images and covers CUDA 11.8 -> 13.0.
# Run from the repository root:  ./packaging/run_all_packaging.sh
set -eo pipefail

VERSION="$(cat src/python/VERSION.txt)"
OUT="$(pwd)/wheelhouse"
mkdir -p "$OUT"

# Each row: CUDA_VERSION  CUDA_TAG  UBUNTU        FAISS_TAG  FAISS_ARCHS
# FAISS is built for Turing..Hopper; the trailing plain "90" also emits PTX so
# FAISS JITs onto Blackwell (sm_100/sm_120), while tsne-cuda's own kernels get
# native Blackwell cubins from its CUDA>=12.8 gencode ladder.
MATRIX=(
  "11.8.0  cu118  ubuntu22.04  v1.9.0  70;75;80;86;89;90"
  "12.4.1  cu124  ubuntu22.04  v1.9.0  75;80;86;89;90"
  "12.6.3  cu126  ubuntu22.04  v1.9.0  75;80;86;89;90"
  "12.9.1  cu129  ubuntu22.04  v1.9.0  75;80;86;89;90"
  "13.0.1  cu130  ubuntu24.04  v1.11.0 75;80;86;89;90"
)

build_one() {
  read -r CUDA_VERSION CUDA_TAG UBUNTU FAISS_TAG FAISS_ARCHS <<<"$1"
  echo ">>> building wheel for CUDA ${CUDA_VERSION} (${CUDA_TAG})"
  # --network=host + the legacy builder (DOCKER_BUILDKIT=0): some Docker setups
  # have no DNS/egress on the default bridge during build steps (apt/git/wget),
  # and BuildKit does not always honor host networking; the legacy builder does.
  DOCKER_BUILDKIT=0 docker build --network=host -f packaging/Dockerfile \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg UBUNTU="${UBUNTU}" \
    --build-arg CUDA_TAG="${CUDA_TAG}" \
    --build-arg FAISS_TAG="${FAISS_TAG}" \
    --build-arg FAISS_ARCHS="${FAISS_ARCHS}" \
    --build-arg TSNE_VERSION="${VERSION}" \
    -t "tsnecuda-wheel:${CUDA_TAG}" .
  # Extract the wheel baked into the image at /wheels.
  local cid
  cid="$(docker create "tsnecuda-wheel:${CUDA_TAG}")"
  docker cp "${cid}:/wheels/." "${OUT}/"
  docker rm "${cid}" >/dev/null
}

for row in "${MATRIX[@]}"; do
  if ! build_one "$row"; then
    echo "!!! build failed for: $row (continuing)" >&2
  fi
done

echo "=== wheels in ${OUT} ==="
ls -l "$OUT"
