#include "image.h"
#include "math.h"
#include <opencv2/core/matx.hpp>

vector<double> generateGaussianKernel1D(double sigma, int &kernelRadius) {
  if (sigma <= 0) {
    throw std::invalid_argument("Sigma must be positive");
  }

  // get 99% of the distribution.
  kernelRadius = static_cast<int>(ceil(3 * sigma));
  int kernelSize = 2 * kernelRadius + 1;

  vector<double> kernel_1d(kernelSize);
  double sum = 0.0;
  for (int x = -kernelRadius; x <= kernelRadius; x++) {
    double value = exp(-(x * x) / (2 * sigma * sigma));
    kernel_1d[x + kernelRadius] = value;
    sum += value;
  }

  // Normalize the kernel
  for (int i = 0; i < kernelSize; i++) {
    kernel_1d[i] /= sum;
  }

  return kernel_1d;
}

/**
  Manual Gaussian blur implementation without using OpenCV's built-in functions.
  I am leveragin the fact that a Gaussian kernel is separable, so we can apply
  the blur in two passes (horizontal and vertical) for better performance.

  That way we can compare a O(WH*N) algorithm in C++ vs My O(WH*N) in CUDA
*/
Image *gaussianBlurCPU(const Image *inputImage, double sigma) {
  // Ensure the input image is in float format
  cv::Mat inputFloat = inputImage->data;

  int kernelRadius;
  std::vector<double> kernel1D = generateGaussianKernel1D(sigma, kernelRadius);

  int width = inputImage->getWidth();
  int height = inputImage->getHeight();

  // Temporary image for horizontal pass
  cv::Mat tempMat = cv::Mat::zeros(height, width, CV_32FC3);
  Image tempImage(width, height, tempMat);

  // --- Horizontal pass ---
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      cv::Vec3f pixelValue(0, 0, 0);
      for (int k = -kernelRadius; k <= kernelRadius; k++) {
        int imgX = std::clamp(x + k, 0, width - 1);
        cv::Vec3f neighborPixel;
        inputImage->getPixel(y, imgX, neighborPixel); // <-- fixed row/col
        for (int c = 0; c < 3; c++)
          pixelValue[c] +=
              neighborPixel[c] * static_cast<float>(kernel1D[k + kernelRadius]);
      }
      tempImage.setPixel(y, x, pixelValue);
    }
  }

  cv::Mat outputMat = cv::Mat::zeros(height, width, CV_32FC3);
  Image *outputImage = new Image(width, height, outputMat);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      cv::Vec3f pixelValue(0, 0, 0);
      for (int k = -kernelRadius; k <= kernelRadius; k++) {
        int imgY = std::clamp(y + k, 0, height - 1);
        cv::Vec3f neighborPixel;
        tempImage.getPixel(imgY, x, neighborPixel);
        for (int c = 0; c < 3; c++)
          pixelValue[c] +=
              neighborPixel[c] * static_cast<float>(kernel1D[k + kernelRadius]);
      }
      outputImage->setPixel(y, x, pixelValue);
    }
  }

  return outputImage;
}
