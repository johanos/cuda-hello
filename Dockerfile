# Base image: CUDA 13 + Ubuntu 24.04
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

# Install build tools
RUN apt-get update && apt-get install -y \
  build-essential \
  clang  \
  clangd \
  clang-format \
  cmake \
  gdb \
  make \
  git \
  curl \
  ca-certificates \
  &&  rm -rf /var/lib/apt/lists/*

# set clang as the default C/C++ compiler
RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 \
  && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100 \
  && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Compile CUDA program
RUN nvcc hello.cu -o hello
RUN nvcc simple_add.cu -o simple_add

# Default command
CMD ["./simple_add"]
