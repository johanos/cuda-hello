cmake_minimum_required(VERSION 3.20)
project(my_cuda_project LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# Target RTX 50xx GPU
set(CMAKE_CUDA_ARCHITECTURES 90)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Add CUDA include directories

add_executable(hello hello.cu)