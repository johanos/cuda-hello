#include "image.h"
#include <string>
using namespace std;

Image::Image(int width, int height) : width(width), height(height) {
    data = new unsigned char[width * height * 4]; // RGBA
}

Image::~Image() {
    delete[] data;
}


int Image::getWidth() const {
    return width;
}

int Image::getHeight() const {
    return height;
}


Image* Image::loadFromFile(const string *filename) {
    // Placeholder for image loading logic
    return nullptr;
}