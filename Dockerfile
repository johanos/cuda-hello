# Base image: CUDA 13 + Ubuntu 24.04
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

# Install build tools
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake 



# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Compile CUDA program
RUN nvcc hello.cu -o hello
RUN nvcc simple_add.cu -o simple_add

# Default command
CMD ["./simple_add"]
