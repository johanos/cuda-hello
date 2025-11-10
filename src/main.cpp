#include "image.h"
#include <GLFW/glfw3.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

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

int main() {
  // Load the image
  Image *inputImage = Image::loadFromFile("./images/trudy.jpg");
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

  // Display the final blurred result (using sigma = 2.0)
  Image *finalResult = gaussianBlur(inputImage, 2.0);
  float aspectRatio =
      static_cast<float>(finalResult->getWidth()) / finalResult->getHeight();
  cv::Size windowSize(800, static_cast<int>(800 / aspectRatio));

  cv::Mat resized;
  cv::resize(finalResult->data, resized, windowSize, 0, 0, cv::INTER_AREA);
  cv::namedWindow("Blurred Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Blurred Image", resized);

  delete finalResult;
  cv::waitKey(0); // Wait for a key press to close the window  // Clean up
  delete inputImage;
  return 0;
}
