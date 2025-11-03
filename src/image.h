#ifndef IMAGE_H
#define IMAGE_H
#include <string>
using namespace std;

class Image {
  private:
    int width;
    int height;
    unsigned char* data;
public:
    Image(int width, int height);
    ~Image();

    void setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b);
    void getPixel(int x, int y, unsigned char &r, unsigned char &g, unsigned char &b) const;

    int getWidth() const;
    int getHeight() const;

    static Image* loadFromFile(const string* filename);
    void saveToFile(const string* filename) const;
};
#endif // IMAGE_H