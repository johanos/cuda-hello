cmake_minimum_required(VERSION 3.20)
project(my_cuda_project LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# Target RTX 50xx GPU
set(CMAKE_CUDA_ARCHITECTURES 90)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(gaussian_blur src/main.cpp)

target_link_libraries(gaussian_blur 
    PRIVATE 
    OpenGL::GL
    glfw
)