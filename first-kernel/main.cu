#include <cuda.h>
#include <iostream>

// Sorta following introductory series by Creel
// (https://youtube.com/@WhatsACreel)

__global__ void AddIntsCUDA(int *a, int *b) {
  // Dumb kernel that wastes a GPU thread!
  a[0] += b[0];
}

// Nice way to use cudaMalloc when experimenting with CUDA
//
// Idk if stuff like this is considered good or bad practice.
template <typename T> T *tryCudaMalloc() {
  T *ptr;
  if (cudaMalloc(&ptr, sizeof(T)) != cudaSuccess) {
    std::cerr << "cudaMalloc failed" << std::endl;
    return nullptr;
  }
  return ptr;
}

int main() {
  int a = 5, b = 9;

  int *d_a = tryCudaMalloc<int>();
  int *d_b = tryCudaMalloc<int>();

  if (d_a == nullptr || d_b == nullptr) {
    return 1;
  }

  cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

  std::cout << "a old = " << a << std::endl;

  AddIntsCUDA<<<1, 1>>>(d_a, d_b);

  cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "a new = " << a << std::endl;

  cudaFree(d_a);
  cudaFree(d_b);

  std::cout << "CUDA version: " << CUDA_VERSION << std::endl;
  return 0;
}
