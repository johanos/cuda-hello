#include "gaussian_blur_cpu.h"
#include "image.h"
#include <cstddef>
#include <cstring>

struct ImageObject {
  float *data; // Pointer to image data in device memory
  int width;
  int height;
};

struct Kernel {
  double *data; // Pointer to kernel data in device memory
  int radius;
};

/**
  I will do a separable Gaussian blur so we will apply a horizontal pass
  followed by a vertical pass

  This takes in the device pointers to the inputImage, outputImage, and the
  Kernel plus other parameters like width, height, and kernel radius
*/
__global__ void horizontalPass(const ImageObject in, ImageObject out,
                               const Kernel kernel) {

  int width = in.width;
  int height = in.height;
  // Calculate pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int kernelRadius = kernel.radius;
  // Apply horizontal Gaussian blur
  float sum[3] = {0.0f, 0.0f, 0.0f};
  for (int k = -kernelRadius; k <= kernelRadius; ++k) {
    int sampleX = min(max(x + k, 0), width - 1);
    int idx = (y * width + sampleX) * 3;
    sum[0] += in.data[idx] * kernel.data[k + kernelRadius];
    sum[1] += in.data[idx + 1] * kernel.data[k + kernelRadius];
    sum[2] += in.data[idx + 2] * kernel.data[k + kernelRadius];
  }

  int outIdx = (y * width + x) * 3;
  out.data[outIdx] = sum[0];
  out.data[outIdx + 1] = sum[1];
  out.data[outIdx + 2] = sum[2];
}

__global__ void verticalPass(const ImageObject in, ImageObject out,
                             const Kernel kernel) {
  int width = in.width;
  int height = in.height;
  // Calculate pixel coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int kernelRadius = kernel.radius;
  // Apply vertical Gaussian blur
  float sum[3] = {0.0f, 0.0f, 0.0f};
  for (int k = -kernelRadius; k <= kernelRadius; ++k) {
    int sampleY = min(max(y + k, 0), height - 1);
    int idx = (sampleY * width + x) * 3;
    sum[0] += in.data[idx] * kernel.data[k + kernelRadius];
    sum[1] += in.data[idx + 1] * kernel.data[k + kernelRadius];
    sum[2] += in.data[idx + 2] * kernel.data[k + kernelRadius];
  }

  int outIdx = (y * width + x) * 3;
  out.data[outIdx] = sum[0];
  out.data[outIdx + 1] = sum[1];
  out.data[outIdx + 2] = sum[2];
}

/**
  This will convert a host generated Gaussian Kernel to a CUDA unified memory
  device pointer
 */
double *generateGaussianKernelMemory(const double sigma, int &kernelRadius) {
  vector<double> kernel = generateGaussianKernel1D(sigma, kernelRadius);

  double *kernelDevice = nullptr;
  size_t kernelSize = kernel.size() * sizeof(double);
  // Allocate unified memory
  cudaError_t err = cudaMallocManaged(&kernelDevice, kernelSize);
  if (err != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    return nullptr;
  }

  // Copy host vector to unified memory
  // kernel is std::vector<double> and kernelDevice is double*
  // it is a raw pointer to unified memory
  std::copy(kernel.begin(), kernel.end(), kernelDevice);

  return kernelDevice;
}

/**
  We are relying on the fact that CV::Mat is stored in a continuous block of
  memory so we can directly get the image data pointer and copy it to the
  device memory if this wasn't the case we would have to copy elements
 */
float *generateInputImageMemory(const Image *inputImage) {

  // is the OpenCV Mat data that powers the Image class
  cv::Mat imgFloat = inputImage->data;
  if (!imgFloat.isContinuous()) {
    std::cerr << "Image data is not continuous in memory!" << std::endl;
    return nullptr;
  }

  if (imgFloat.type() != CV_32FC3) {
    std::cerr << "Image data is not of type CV_32FC3!" << std::endl;
    return nullptr;
  }

  float *inputDevice = nullptr;
  cudaError_t err =
      cudaMallocManaged(&inputDevice, imgFloat.total() * imgFloat.elemSize());

  if (err != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    return nullptr;
  }
  memcpy(inputDevice, imgFloat.ptr<float>(0),
         imgFloat.total() * imgFloat.elemSize());

  return inputDevice;
}

Image *convertDeviceToImage(float *deviceData, int width, int height) {
  // Create a new Image object
  cv::Mat imgFloat(height, width, CV_32FC3);
  Image *outputImage = new Image(width, height, imgFloat);

  memcpy(imgFloat.ptr<float>(0), deviceData,
         imgFloat.total() * imgFloat.elemSize());

  return outputImage;
}

Image *gaussianBlurCUDA(const Image *inputImage, const double sigma) {

  int kernelRadius;
  double *kernelDevice = generateGaussianKernelMemory(sigma, kernelRadius);
  float *inputImageDevice = generateInputImageMemory(inputImage);
  float *tempDevice = nullptr;
  float *outputImageDevice = nullptr;

  if (!kernelDevice || !inputImageDevice) {
    return nullptr;
  }

  dim3 blockSize(32, 32);
  dim3 gridSize((inputImage->getWidth() + blockSize.x - 1) / blockSize.x,
                (inputImage->getHeight() + blockSize.y - 1) / blockSize.y);

  // Allocate output image memory
  cudaError_t err = cudaMallocManaged(
      &outputImageDevice,
      inputImage->getWidth() * inputImage->getHeight() * 3 * sizeof(float));

  if (err != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    cudaFree(kernelDevice);
    cudaFree(inputImageDevice);
    return nullptr;
  }

  err = cudaMallocManaged(&tempDevice, inputImage->getWidth() *
                                           inputImage->getHeight() * 3 *
                                           sizeof(float));

  if (err != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    cudaFree(kernelDevice);
    cudaFree(inputImageDevice);
    cudaFree(outputImageDevice);
    return nullptr;
  }

  // Create ImageObject structs for input and output images
  ImageObject inputObj = {inputImageDevice, inputImage->getWidth(),
                          inputImage->getHeight()};
  ImageObject tempObj = {tempDevice, inputImage->getWidth(),
                         inputImage->getHeight()};
  ImageObject outputObj = {outputImageDevice, inputImage->getWidth(),
                           inputImage->getHeight()};
  Kernel kernelObj = {kernelDevice, kernelRadius};

  // Launch horizontal pass
  horizontalPass<<<gridSize, blockSize>>>(inputObj, tempObj, kernelObj);
  cudaDeviceSynchronize();
  // Launch vertical pass
  verticalPass<<<gridSize, blockSize>>>(tempObj, outputObj, kernelObj);
  cudaDeviceSynchronize();

  // Now that we have the device pointer for the output image data we gotta
  // repackage it to return
  Image *res = convertDeviceToImage(outputImageDevice, inputImage->getWidth(),
                                    inputImage->getHeight());
  cudaFree(outputImageDevice);
  cudaFree(inputImageDevice);
  cudaFree(kernelDevice);
  cudaFree(tempDevice);
  return res;
}