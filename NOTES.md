# CUDA Notes

* Kernels always return void
  * Signature always starts with `__global__ void name` or `__device__ void name`

* CUDA functions return a result enum. Check for `cudaSuccess` to handle errors.

## Specifiers

* `__global__`
  * Kernel that that runs on the GPU but is called from the CPU

* `__device__`
  * GPU-only function that can be called from within a kernel

* `__host__`
  * Normal, CPU-only function that cannot be called from within a kernel

* The behavior of `__host__` is assumed when no specifier is given

## Basic functions

* `cudaMalloc(void **devicePtr, size_t sizeInBytes)`
  Reserves memory on the device.

* `cudaFree(void **devicePtr)`
  Frees memory on the device.

* `cudaMemcpy(void *destination, void *source, size_t sizeInBytes, enum direction)`
  Copies data from host to device.

  ```cuda
  enum direction {
    cudaMemcpyDeviceToHost
    cudaMemcpyHostToDevice
  }
  ```
