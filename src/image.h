#ifndef IMAGE_H
#define IMAGE_H
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

class Image {
private:
  int width;
  int height;

public:
  cv::Mat data;
  Image(int width, int height, cv::Mat data);
  ~Image();

  template <typename T> void setPixel(int row, int col, const T &color) {
    data.at<T>(row, col) = color;
  }

  template <typename T> void getPixel(int row, int col, T &color) const {
    color = data.at<T>(row, col);
  }

  int getWidth() const;
  int getHeight() const;

  static Image *loadFromFile(const string &filename);
  void saveToFile(const string *filename) const;
};
#endif // IMAGE_H