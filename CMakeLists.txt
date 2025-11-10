cmake_minimum_required(VERSION 3.20)
project(my_cuda_project LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Copy images directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/images DESTINATION ${CMAKE_BINARY_DIR})

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
find_package(OpenCV REQUIRED)

add_executable(gaussian_blur 
    src/main.cpp
    src/image.cpp
)

# Manually add the base include path for OpenCV 4 on Linux/Unix systems.
# This ensures that internal includes like #include "opencv2/opencv_modules.hpp"
# resolve correctly when the include path is /usr/include/opencv4.
target_include_directories(gaussian_blur PRIVATE /usr/include/opencv4)

# Prefer imported OpenCV targets when available, otherwise fall back to
# ${OpenCV_INCLUDE_DIRS} / ${OpenCV_LIBS} provided by the FindOpenCV module.
if(TARGET OpenCV::opencv_core)
    target_link_libraries(gaussian_blur
    PRIVATE
    OpenGL::GL
    glfw
    OpenCV::opencv_core
    OpenCV::opencv_imgproc
    OpenCV::opencv_highgui
  )
else()
  target_include_directories(gaussian_blur PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(gaussian_blur
    PRIVATE
    OpenGL::GL
    glfw
    ${OpenCV_LIBS}
  )
endif()