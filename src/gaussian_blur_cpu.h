#ifndef GAUSSIAN_BLUR_CPU_H
#define GAUSSIAN_BLUR_CPU_H
#include "image.h"

Image *gaussianBlurCPU(const Image *inputImage, double sigma);
vector<double> generateGaussianKernel1D(double sigma, int &kernelRadius);
#endif