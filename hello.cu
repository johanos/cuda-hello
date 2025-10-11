#include <iostream>

__global__ void helloGPU()
{
  printf("Hello World from GPU! Thread %d\n", threadIdx.x);
}

int main()
{
  // Launch 1 block of 5 threads
  helloGPU<<<1, 20>>>();

  // Wait for GPU to finish
  cudaDeviceSynchronize();

  std::cout << "Hello World from CPU!" << std::endl;
  return 0;
}
