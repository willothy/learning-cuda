// Embarrassingly parallel algorithms

#include <chrono>
#include <ctime>
#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

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

static inline float distance_no_sqrt(float3 a, float3 b) {
  return (((a.x - b.x) * (a.x - b.x)) + ((a.y - b.y) * (a.y - b.y)) +
          ((a.z - b.z) * (a.z - b.z)));
}

namespace NearestNeighbors {
// For a list of 3D points, find the nearest neighbor of each point.

void Unaccelerated(float3 *points, int *indices, int count) {
  if (count <= 1) {
    return;
  }
  for (int current = 0; current < count; current++) {
    float distToClosest = 3.40282e38f;
    for (int i = 0; i < count; i++) {
      if (i == current)
        continue;
      float dist = distance_no_sqrt(points[current], points[i]);
      if (dist < distToClosest) {
        distToClosest = dist;
        indices[current] = i;
      }
    }
  }
}

__global__ void AcceleratedInternal(float3 *points, int *indices, int count) {
  if (count <= 1) {
    return;
  }

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= count) {
    return;
  }

  float3 thisPoint = points[idx];
  float distToClosest = 3.40282e38f;
  for (int i = 0; i < count; i++) {
    if (i == idx) {
      continue;
    }
    float dist = (thisPoint.x - points[i].x) * (thisPoint.x - points[i].x);
    dist += (thisPoint.y - points[i].y) * (thisPoint.y - points[i].y);
    dist += (thisPoint.z - points[i].z) * (thisPoint.z - points[i].z);
    if (dist < distToClosest) {
      distToClosest = dist;
      indices[idx] = i;
    }
  }
}

void Accelerated(float3 *points, int *indices, int count) {
  if (count <= 1) {
    return;
  }

  float3 *d_points = tryCudaMalloc<float3>(count);
  int *d_indices = tryCudaMalloc<int>(count);
  if (d_points == nullptr || d_indices == nullptr) {
    std::cerr << "cudaMalloc failed" << std::endl;
    return;
  }

  cudaMemcpy(d_points, points, sizeof(float3) * count, cudaMemcpyHostToDevice);
  cudaMemcpy(d_indices, indices, sizeof(int) * count, cudaMemcpyHostToDevice);

  AcceleratedInternal<<<(count / 32) + 1, 32>>>(d_points, d_indices, count);

  cudaMemcpy(indices, d_indices, sizeof(int) * count, cudaMemcpyDeviceToHost);

  cudaFree(d_points);
  cudaFree(d_indices);
}

void GenerateRandomPoints(float3 *points, int count) {
  for (int i = 0; i < count; i++) {
    points[i].x = (float)((rand() % 10000) - 5000);
    points[i].y = (float)((rand() % 10000) - 5000);
    points[i].z = (float)((rand() % 10000) - 5000);
  }
}

void Bench(const char *name, void (*fn_to_benchmark)(float3 *, int *, int),
           float3 *points, int n_points, int n_trials,
           std::vector<int *> *indices) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  int count = n_points;

  int *indexOfClosest = new int[count];

  duration<double, std::milli> fastest =
      duration_cast<milliseconds>(std::chrono::hours(1));
  duration<double, std::milli> average =
      duration_cast<milliseconds>(std::chrono::milliseconds(0));

  std::cout << "Running benchmark: " << name << std::endl;

  auto start = high_resolution_clock::now();

  for (int i = 0; i < n_trials; i++) {
    auto t1 = high_resolution_clock::now();
    fn_to_benchmark(points, indexOfClosest, count);
    auto t2 = high_resolution_clock::now();

    auto result = duration_cast<milliseconds>(t2 - t1);
    if (result < fastest) {
      fastest = result;
    }
    average += result;
  }

  auto finish = high_resolution_clock::now();
  auto elapsed = duration_cast<milliseconds>(finish - start);

  average /= n_trials;

  indices->push_back(indexOfClosest);

  std::cout << "Number of points: " << n_points << std::endl;
  std::cout << "Trials: " << n_trials << std::endl;
  std::cout << "Total time: " << elapsed.count() << "ms" << std::endl;
  std::cout << "Fastest: " << fastest.count() << "ms" << std::endl;
  std::cout << "Average: " << average.count() << "ms" << std::endl;
  std::cout << std::endl;
}

void Run() {
#define N_POINTS 10000
#define N_TRIALS 10
  float3 points[N_POINTS] = {0};

  GenerateRandomPoints(points, N_POINTS);

  std::vector<int *> indices;

  Bench("Unaccelerated", Unaccelerated, points, N_POINTS, N_TRIALS, &indices);
  Bench("Accelerated", Accelerated, points, N_POINTS, N_TRIALS, &indices);

  for (int i = 0; i < N_POINTS; i++) {
    if (indices[0][i] != indices[1][i]) {
      std::cerr << "Mismatch at index " << i << std::endl;
      std::cerr << "Unaccelerated: " << indices[0][i] << std::endl;
      std::cerr << "Accelerated: " << indices[1][i] << std::endl;
    }
  }

  for (int i = 0; i < indices.size(); i++) {
    delete[] indices[i];
  }

#undef N_POINTS
#undef N_TRIALS
}
} // namespace NearestNeighbors

int main() {
  NearestNeighbors::Run();

  return 0;
}
