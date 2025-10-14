#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__global__ void helloGPU() {
  printf("Hello World from GPU! Thread %d\n", threadIdx.x);
}

int main() {

  // Launch 1 block of 5 threads
  helloGPU<<<1, 20>>>();

  cudaDeviceSynchronize();

  // Wait for GPU to finish
  cout << "Hello World from CPU!" << endl;
  return 0;
}