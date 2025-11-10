#include "image.h"
#include <iostream>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
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

void Image::setPixel(int row, int col, cv::Vec3b color) {
  data.at<cv::Vec3b>(row, col) = color;
}

void Image::getPixel(int row, int col, cv::Vec3b &color) const {
  color = data.at<cv::Vec3b>(row, col);
}

Image *Image::loadFromFile(const string &filename) {
  // Load as 3-channel BGR (ignores any alpha), then convert to RGB.
  Mat img = cv::imread(filename, cv::IMREAD_COLOR);
  if (img.empty()) {
    cerr << "Failed to load image: " << filename << endl;
    return nullptr;
  }

  return new Image(img.cols, img.rows, img);
}
