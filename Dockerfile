# Base image: CUDA 12.9 + Ubuntu 24.04
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

# Install build tools, clang, and set up NVIDIA repo for Nsight Systems
RUN apt-get update && apt-get install -y wget gnupg ca-certificates \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i --force-overwrite cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && rm -f /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && apt-get install -y \
    build-essential \
    clang \
    clangd \
    clang-format \
    cmake \
    gdb \
    make \
    git \
    curl \
    wget \
    nsight-systems-2025.3.2 \
    # OpenGL dependencies
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    libglew-dev \
    libglfw3-dev \
    libglm-dev \
    pkg-config \
    xorg-dev \
    && rm -rf /var/lib/apt/lists/*

# Add Nsight Systems to PATH
ENV PATH=/opt/nvidia/nsight-systems/2025.3.2/bin:$PATH

# Set clang as default C/C++ compiler
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100

# Set working directory
WORKDIR /workspace

# Add /root to PATH
ENV PATH="/root:${PATH}"

# WSL-specific environment setup
ENV WAYLAND_DISPLAY=wayland-0
ENV XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
ENV PULSE_SERVER=/mnt/wslg/PulseServer
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
