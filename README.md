# CUDA Image Processing Project (Title TBD)

This repository contains a small project for running CUDA code that performs an image processing task (inference/transform/analysis). The intent is to provide a simple, GPU-accelerated pipeline for experimenting with CUDA kernels and image IO.

Note: this project is intended to be run inside a VS Code dev container — in my setup I'm using WSL 2 with a VS Code dev container.

The Dockerfile and devcontainer.json are set up to provide a consistent environment with the necessary CUDA toolkit and libraries. VSCode _should_ install the things you need.

## Getting started

Prerequisites:
- NVIDIA GPU with a supported driver
- CUDA Toolkit installed
- CMake and a C/C++ compiler

Build (typical):
```bash
./build_project.sh
# or manually:
cmake -S . -B build
cd build
make
```

## Project layout (expected)
- src/         — CUDA and host-side source files
- include/     — headers
- images/      — sample images for testing
- CMakeLists.txt

## Notes
- Check src/ and CMakeLists.txt for exact build and run targets.
- Ensure the CUDA toolkit version matches the project's requirements.

License: see LICENSE (if present).