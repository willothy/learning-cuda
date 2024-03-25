# CUDA Notes

My own personal CUDA documentation.

## Architecture

### Threads

A thread is a single execution unit that runs a kernel on the GPU.
Similar to CPU threads, but there are often many more of them.

* Threads are cheap on the GPU. Where something would be iterative on the CPU,
  it can be massively parallelized on the GPU.
* Sometimes drawn in NVIDIA docs as a squiggly arrow.

### Thread Blocks

A thread block is a collection of threads, similar to the thread pool
concept but used in a more structured way.

* Collection of threads
* All threads in any single thread block can communicate
* Can be 3-dimensional

### Grid

A kernel is launched as a collection of thread blocks called a Grid.

* This is the "top-level" datastructure of GPU compute
* Can be 3-dimensional

### Kernel

A kernel is a program that runs on the GPU.

### Memory

## API

* Kernels always return void
  * Signature always starts with `__global__ void name` or `__device__ void name`

* CUDA functions return a result enum. Check for `cudaSuccess` to handle errors.

### Specifiers

* `__global__`
  * Kernel that that runs on the GPU but is called from the CPU

* `__device__`
  * GPU-only function that can be called from within a kernel

* `__host__`
  * Normal, CPU-only function that cannot be called from within a kernel

* The behavior of `__host__` is assumed when no specifier is given

### "Launch"-ing a kernel

Kernels are "launched" with the `<<<>>>` syntax.

The first launch parameter is the number of blocks to use.
The second is the number of threads to use for each block.

These parameters are called the "launch configuration."

```cuda
SomeKernel<<<N_BLOCKS, N_THREADS>>>(...);
```

* Maximum 1024 threads per block (512 for some older cards)
* Can launch 2^32 - 1 blocks in a single launch, or 2^16 - 1 on some older cards

### Basic functions

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

### Data Structures

#### `dim3`

3-component vector type.

* `dim3 threads(256)`
  * Initialize x as 256, y and z will both be 1

* `dim3 blocks(100, 100)`
  * Initialize x and y, z will be 1

### Thread-local variables

Each thread runs individually and has access to some information about itself.

All are `dim3` structures.

* `threadIdx`: Thread index within the block
* `blockIdx`: Block index within the grid
* `blockDim`: Block dimensions in threads
* `gridDim`: Grid dimensions in blocks

#### Calculating a unique thread id

It is common for a kernel to calculate a unique id for each thread. Each thread can calculate its unique id with something like this:

```cuda
int id = blockIdx.x * blockDim.x + threadIdx.x;
```
