#include "image.h"
#include <iostream>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/hal/interface.h>
#include <string>
using namespace cv;

Image::Image(int width, int height, Mat data) : width(width), height(height) {
  this->data = data;
}

int Image::getWidth() const { return width; }

int Image::getHeight() const { return height; }

Image::~Image() {
  // cv::Mat handles its own memory management
}

/**
  Load an image from file into an Image object, it is loaded as BGR Float32
*/
Image *Image::loadFromFile(const string &filename) {
  // Load as 3-channel BGR (ignores any alpha), then convert to RGB.
  Mat img = cv::imread(filename, cv::IMREAD_COLOR);
  // Convert to float32 and normalize to [0,1]
  img.convertTo(img, CV_32FC3, 1.0 / 255.0);
  if (img.empty()) {
    cerr << "Failed to load image: " << filename << endl;
    return nullptr;
  }

  return new Image(img.cols, img.rows, img);
}
