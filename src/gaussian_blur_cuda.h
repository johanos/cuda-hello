#ifndef GAUSSIAN_BLUR_CUDA_H
#define GAUSSIAN_BLUR_CUDA_H
#include "image.h"

Image *gaussianBlurCUDA(const Image *inputImage, double sigma);

#endif