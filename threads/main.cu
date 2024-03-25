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

__global__ void AddInts(int *a, int *b, int count) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < count) {
    a[id] += b[id];
  }
}

int main() {
  srand(time(NULL));

  int count = 1000;
  int *h_a = new int[count];
  int *h_b = new int[count];

  for (int i = 0; i < count; i++) {
    h_a[i] = rand() % 1000;
    h_b[i] = rand() % 1000;
  }

  std::cout << "Before" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << h_a[i] << " " << h_b[i] << std::endl;
  }

  int *d_a = tryCudaMalloc<int>(count);
  int *d_b = tryCudaMalloc<int>(count);
  if (d_a == nullptr || d_b == nullptr) {
    if (d_a != nullptr) {
      cudaFree(d_a);
    }
    if (d_b != nullptr) {
      cudaFree(d_b);
    }
    return 1;
  }

#define FREE_ALL()                                                             \
  do {                                                                         \
    cudaFree(d_a);                                                             \
    cudaFree(d_b);                                                             \
    delete h_a;                                                                \
    delete h_b;                                                                \
  } while (0)

  if (cudaMemcpy(d_a, h_a, sizeof(int) * count, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cout << "Memcpy failed" << std::endl;
    FREE_ALL();
    return 1;
  }
  if (cudaMemcpy(d_b, h_b, sizeof(int) * count, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::cout << "Memcpy failed" << std::endl;
    FREE_ALL();
    return 1;
  }

  AddInts<<<count / 256 + 1, 256>>>(d_a, d_b, count);

  if (cudaMemcpy(h_a, d_a, sizeof(int) * count, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    std::cout << "Memcpy failed" << std::endl;
    FREE_ALL();
    return 1;
  }

  std::cout << "After" << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << h_a[i] << " " << h_b[i] << std::endl;
  }

  FREE_ALL();

  return 0;
}
