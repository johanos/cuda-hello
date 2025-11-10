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

  void setPixel(int row, int col, cv::Vec3b color);
  void getPixel(int row, int col, cv::Vec3b &color) const;

  int getWidth() const;
  int getHeight() const;

  static Image *loadFromFile(const string &filename);
  void saveToFile(const string *filename) const;
};
#endif // IMAGE_H