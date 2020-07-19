# This sets up a docker file which can be used for the conda build of the TSNE-Cuda
# repository.

# Base the build file on ubuntu
FROM ubuntu:16.04
USER root
WORKDIR /home/root

#### Install new CUDA versions ####
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
    rm -rf /var/lib/apt/lists/* && \
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list


# CUDA 11
ENV CUDA_VERSION 11.0.2
ENV CUDA_PKG_VERSION 11-0=$CUDA_VERSION-1
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-11-0

ENV NCCL_VERSION 2.7.6
RUN apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda11.0 && \
    apt-mark hold libnccl2

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda11.0


# CUDA 10.2
ENV CUDA_VERSION 10.2.89
ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-2

ENV NCCL_VERSION 2.7.6
RUN apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda10.2 && \
    apt-mark hold libnccl2

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.2


# CUDA 10.1
ENV CUDA_VERSION 10.1.168
ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-1

ENV NCCL_VERSION 2.4.2
RUN apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda10.1 && \
    apt-mark hold libnccl2

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.1

# CUDA 10.0
ENV CUDA_VERSION 10.0.130
ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1
RUN apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-0

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION

# CUDA 9.2
ENV CUDA_VERSION 9.2.148
ENV CUDA_PKG_VERSION 9-2=$CUDA_VERSION-1
RUN apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION

# CUDA 9.1
ENV CUDA_VERSION 9.1.85
ENV CUDA_PKG_VERSION 9-1=$CUDA_VERSION-1
RUN apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION

# CUDA 9.0
ENV CUDA_VERSION 9.0.176
ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1
RUN apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-cublas-9-0=9.0.176.4-1

RUN apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-core-9-0=9.0.176.3-1 \
        cuda-cublas-dev-9-0=9.0.176.4-1
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# CUDA 8.0
ENV CUDA_VERSION 8.0.61
ENV CUDA_PKG_VERSION 8-0=$CUDA_VERSION-1
RUN apt-get install -y --no-install-recommends \
        cuda-nvrtc-$CUDA_PKG_VERSION \
        cuda-nvgraph-$CUDA_PKG_VERSION \
        cuda-cusolver-$CUDA_PKG_VERSION \
        cuda-cublas-8-0=8.0.61.2-1 \
        cuda-cufft-$CUDA_PKG_VERSION \
        cuda-curand-$CUDA_PKG_VERSION \
        cuda-cusparse-$CUDA_PKG_VERSION \
        cuda-npp-$CUDA_PKG_VERSION \
        cuda-cudart-$CUDA_PKG_VERSION

RUN apt-get install -y --no-install-recommends \
        cuda-core-$CUDA_PKG_VERSION \
        cuda-misc-headers-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-nvrtc-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-nvgraph-dev-$CUDA_PKG_VERSION \
        cuda-cusolver-dev-$CUDA_PKG_VERSION \
        cuda-cublas-dev-8-0=8.0.61.2-1 \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-curand-dev-$CUDA_PKG_VERSION \
        cuda-cusparse-dev-$CUDA_PKG_VERSION \
        cuda-npp-dev-$CUDA_PKG_VERSION \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-driver-dev-$CUDA_PKG_VERSION

# Clean up after CUDA install
RUN rm -rf /var/lib/apt/lists/*

# Install useful packages
RUN apt-get update
RUN apt-get install curl git -y --no-install-recommends

# Install anaconda
RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash miniconda.sh -b -p /home/root/miniconda && rm miniconda.sh
RUN /home/root/miniconda/bin/conda install conda-build
ENV PATH $PATH:/home/root/miniconda/bin/

# Copy the build script
COPY clone_and_build.sh ./
RUN chmod +x ./clone_and_build.sh

# Setup the commands
CMD ./clone_and_build.sh
