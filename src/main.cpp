#include "gaussian_blur_cpu.h"
#include "gaussian_blur_cuda.h"
#include "image.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/matx.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

struct BlurTimes {
  double controlMs;
  double cpuMs;
  double cudaMs;
};

/**
    Manual Gaussian blur implementation using vanilla C++ and OpenCV.
*/
Image *gaussianBlur(const Image *inputImage, double sigma) {
  // Create an output image with the same dimensions
  cv::Mat outputMat;
  cv::GaussianBlur(inputImage->data, outputMat, cv::Size(0, 0), sigma);

  return new Image(inputImage->getWidth(), inputImage->getHeight(), outputMat);
}

void profileGaussianBlur(const Image *inputImage, const vector<double> &sigmas,
                         int num_iterations = 10) {
  cout << "\nProfiling Gaussian Blur with " << num_iterations
       << " iterations per sigma value:" << endl;
  cout << setw(10) << "Sigma" << setw(15) << "Avg Time (µs)" << setw(15)
       << "Min Time (µs)" << setw(15) << "Max Time (µs)" << endl;
  cout << string(55, '-') << endl;

  for (double sigma : sigmas) {
    // Warm up run
    Image *result = gaussianBlur(inputImage, sigma);
    delete result;

    // Timing runs
    vector<double> times;
    times.reserve(num_iterations);

    for (int i = 0; i < num_iterations; i++) {
      auto start = high_resolution_clock::now();
      Image *result = gaussianBlur(inputImage, sigma);
      auto end = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      times.push_back(duration.count());
      delete result;
    }

    // Calculate statistics
    double avg_time = 0;
    double min_time = times[0];
    double max_time = times[0];

    for (double t : times) {
      avg_time += t;
      min_time = min(min_time, t);
      max_time = max(max_time, t);
    }
    avg_time /= num_iterations;

    cout << fixed << setprecision(2) << setw(10) << sigma << setw(15)
         << avg_time << setw(15) << min_time << setw(15) << max_time << endl;
  }
  cout << endl;
}

void displayResults(Image *original, Image *blurredCPU, Image *blurredCUDA,
                    const BlurTimes &times) {
  cv::Mat original8U, cpu8U, cuda8U;

  // Convert float images 0-1 → 8-bit for display
  original->data.convertTo(original8U, CV_8UC3, 255.0);
  blurredCPU->data.convertTo(cpu8U, CV_8UC3, 255.0);
  blurredCUDA->data.convertTo(cuda8U, CV_8UC3, 255.0);

  int displayHeight = 400;

  float aspectOriginal = static_cast<float>(original8U.cols) / original8U.rows;
  float aspectCPU = static_cast<float>(cpu8U.cols) / cpu8U.rows;
  float aspectCUDA = static_cast<float>(cuda8U.cols) / cuda8U.rows;

  cv::Mat imgOriginal, imgCPU, imgCUDA;
  cv::resize(original8U, imgOriginal,
             cv::Size(static_cast<int>(displayHeight * aspectOriginal),
                      displayHeight));
  cv::resize(
      cpu8U, imgCPU,
      cv::Size(static_cast<int>(displayHeight * aspectCPU), displayHeight));
  cv::resize(
      cuda8U, imgCUDA,
      cv::Size(static_cast<int>(displayHeight * aspectCUDA), displayHeight));

  int combinedWidth = imgOriginal.cols + imgCPU.cols + imgCUDA.cols;
  int labelHeight = 80; // extra space for timings
  cv::Mat combined(displayHeight + labelHeight, combinedWidth,
                   imgOriginal.type(), cv::Scalar(255, 255, 255));

  imgOriginal.copyTo(
      combined(cv::Rect(0, labelHeight, imgOriginal.cols, imgOriginal.rows)));
  imgCPU.copyTo(combined(
      cv::Rect(imgOriginal.cols, labelHeight, imgCPU.cols, imgCPU.rows)));
  imgCUDA.copyTo(combined(cv::Rect(imgOriginal.cols + imgCPU.cols, labelHeight,
                                   imgCUDA.cols, imgCUDA.rows)));

  // Labels
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 0.7;
  int thickness = 2;
  cv::Scalar color(0, 0, 0);

  cv::putText(combined, "Original", cv::Point(10, 25), fontFace, fontScale,
              color, thickness);
  cv::putText(combined, "CPU Blur", cv::Point(imgOriginal.cols + 10, 25),
              fontFace, fontScale, color, thickness);
  cv::putText(combined, "CUDA Blur",
              cv::Point(imgOriginal.cols + imgCPU.cols + 10, 25), fontFace,
              fontScale, color, thickness);

  // Timings
  cv::putText(combined, "Time: " + std::to_string(times.controlMs) + " ms",
              cv::Point(10, 55), fontFace, 0.6, color, 1);
  cv::putText(combined, "Time: " + std::to_string(times.cpuMs) + " ms",
              cv::Point(imgOriginal.cols + 10, 55), fontFace, 0.6, color, 1);
  cv::putText(combined, "Time: " + std::to_string(times.cudaMs) + " ms",
              cv::Point(imgOriginal.cols + imgCPU.cols + 10, 55), fontFace, 0.6,
              color, 1);

  cv::imshow("Blur Comparison", combined);
  cv::waitKey(0);

  // Also print to console
  std::cout << "Timings (ms):\n";
  std::cout << "Control: " << times.controlMs << "\n";
  std::cout << "CPU Blur: " << times.cpuMs << "\n";
  std::cout << "CUDA Blur: " << times.cudaMs << "\n";
}

// Function to profile your blurs
BlurTimes profileBlurs(Image *inputImage, double sigma, Image *&controlResult,
                       Image *&finalResultCPU, Image *&finalResultCUDA) {
  BlurTimes times;

  // Control blur
  auto start = std::chrono::high_resolution_clock::now();
  controlResult = gaussianBlur(inputImage, sigma);
  auto end = std::chrono::high_resolution_clock::now();
  times.controlMs =
      std::chrono::duration<double, std::milli>(end - start).count();

  // CPU blur
  start = std::chrono::high_resolution_clock::now();
  finalResultCPU = gaussianBlurCPU(inputImage, sigma);
  end = std::chrono::high_resolution_clock::now();
  times.cpuMs = std::chrono::duration<double, std::milli>(end - start).count();

  // CUDA blur
  start = std::chrono::high_resolution_clock::now();
  finalResultCUDA = gaussianBlurCUDA(inputImage, sigma);
  end = std::chrono::high_resolution_clock::now();
  times.cudaMs = std::chrono::duration<double, std::milli>(end - start).count();

  return times;
}

int main() {
  // Load the image
  Image *inputImage = Image::loadFromFile("./images/texture.jpg");
  if (!inputImage) {
    std::cerr << "Failed to load image" << std::endl;
    return -1;
  }

  // Display the original image dimensions
  std::cout << "Loaded image: " << inputImage->getWidth() << "x"
            << inputImage->getHeight() << std::endl;

  // Profile the Gaussian blur with different sigma values
  vector<double> sigmas = {1.0, 2.0, 3.0, 5.0};
  profileGaussianBlur(inputImage, sigmas);

  // Now perform a manual Gaussian blur (without OpenCV) + the one from OpenCV
  // and do a pixel-wise comparison
  double testSigma = 2.0;
  Image *manualBlurredImage = gaussianBlurCPU(inputImage, testSigma);
  Image *opencvBlurredImage = gaussianBlur(inputImage, testSigma);
  // Compare pixel-wise
  int width = inputImage->getWidth();
  int height = inputImage->getHeight();
  double totalDifference = 0.0;
  for (int u = 0; u < height; u++) {
    for (int v = 0; v < width; v++) {
      cv::Vec3b manualPixel;
      cv::Vec3b opencvPixel;
      manualBlurredImage->getPixel(u, v, manualPixel);
      opencvBlurredImage->getPixel(u, v, opencvPixel);
      for (int c = 0; c < 3; c++) {
        totalDifference += abs(static_cast<int>(manualPixel[c]) -
                               static_cast<int>(opencvPixel[c]));
      }
    }
  }
  double avgDifference = totalDifference / (width * height * 3);
  std::cout
      << "Average pixel difference between manual and OpenCV Gaussian blur: "
      << avgDifference << std::endl;

  // Display the final blurred result (using sigma = 2.0)
  Image *control;
  Image *cpu;
  Image *cuda;
  double sigma = 2.0;

  // Profile the blurs
  BlurTimes times = profileBlurs(inputImage, sigma, control, cpu, cuda);
  displayResults(control, cpu, cuda, times);

  delete cpu;
  delete control;
  delete inputImage;
  return 0;
}
