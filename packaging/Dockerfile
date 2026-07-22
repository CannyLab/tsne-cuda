# Copyright (c) 2026 Regents of the University of California
# Released under the MIT License. https://opensource.org/licenses/MIT
#
# Parameterized builder for tsne-cuda Python wheels, one CUDA toolkit per build.
# tsne-cuda 4.0.0 dropped gflags / gtest / CPU-BLAS / MKL and the C++ CLI, so the
# image only needs a C++17 toolchain, CMake, Python + numpy, and a GPU FAISS
# (built from source; FAISS itself still wants an OpenBLAS).
#
# Example:
#   docker build -f packaging/Dockerfile \
#     --build-arg CUDA_VERSION=12.9.1 --build-arg UBUNTU=ubuntu22.04 \
#     --build-arg CUDA_TAG=cu129 --build-arg FAISS_TAG=v1.9.0 \
#     --build-arg FAISS_ARCHS="75;80;86;89;90" -t tsnecuda-wheel:cu129 .
# The finished wheel is baked into the image at /wheels.
ARG CUDA_VERSION=12.9.1
ARG UBUNTU=ubuntu22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU}

# Re-declare build args needed after FROM.
ARG CUDA_TAG=cu129
ARG FAISS_TAG=v1.9.0
ARG FAISS_ARCHS="75;80;86;89;90"
ARG TSNE_VERSION=4.0.0
ARG CMAKE_VERSION=3.28.3
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Force IPv4 (container IPv6 egress is often unroutable) and retry, so apt is
# resilient to transient mirror timeouts.
RUN printf 'Acquire::ForceIPv4 "true";\nAcquire::Retries "8";\n' \
        > /etc/apt/apt.conf.d/99network && \
    apt-get update && apt-get install -y --no-install-recommends \
        git build-essential python3-dev python3-numpy python3-pip \
        wget ca-certificates libopenblas-dev swig && \
    rm -rf /var/lib/apt/lists/*

# tsne-cuda requires CMake >= 3.20; ship a recent one.
RUN wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" && \
    tar -xzf "cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -C /opt && \
    ln -s "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake" /usr/local/bin/cmake && \
    rm "cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"

# Build and install GPU FAISS from source (shared lib + headers into /usr/local).
RUN git clone --depth 1 --branch "${FAISS_TAG}" https://github.com/facebookresearch/faiss.git /faiss && \
    cmake -S /faiss -B /faiss/build -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF \
        -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES="${FAISS_ARCHS}" && \
    cmake --build /faiss/build --target faiss -j"$(nproc)" && \
    cmake --install /faiss/build && ldconfig

# Build the tsne-cuda wheel (no GPU required to compile).
ADD ./ /tsnecuda/
WORKDIR /tsnecuda/build
RUN chmod +x ../packaging/build_and_deploy.sh && \
    ../packaging/build_and_deploy.sh "${CUDA_TAG}" "${TSNE_VERSION}"
