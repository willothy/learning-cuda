#include <ctime>
#include <cuda.h>
#include <iostream>
#include <stdlib.h>

// Nice way to use cudaMalloc when experimenting with CUDA
//
// Idk if stuff like this is considered good or bad practice.
template <typename T> T *tryCudaMalloc(size_t count) {
  T *ptr;
  if (cudaMalloc(&ptr, sizeof(T) * count) != cudaSuccess) {
    std::cerr << "cudaMalloc failed" << std::endl;
    return nullptr;
  }
  return ptr;
}

int main() {
  //

  return 0;
}
