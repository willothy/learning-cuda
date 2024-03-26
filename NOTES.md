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

| Memory    | Access | Scope             | Lifetime   | Speed        |
|-----------|--------|-------------------|------------|--------------|
| Global    | RW     | All threads + CPU | Persistent | Slow, cached |
| Constant  | R      | All threads + CPU | Persistent | Slow, cached |
| Texture   | R      | All threads + CPU | Persistent | Slow, cached |
| Local     | RW     | Thread-local      | Thread     | Slow, cached |
| Shared    | RW     | Block-local       | Block      | Fast         |
| Registers | RW     | Thread-local      | Thread     | Fast         |

#### Global Memory

Global memory is readable and writeable from both the CPU and GPU threads.

* Slower to acceses than shared memory and registers.
* The `cudaMalloc`, `cudaFree`, etc. functions allow working with this memory.
* Every byte is addressable
* Persistent accross kernel calls

#### Local memory

Also part of the main memory of the GPU like global memory, so it's generally "slow."

* Used automatically by the compiler when registers cannot be used (register spilling).
* Arrays that aren't indexed with constants must use local memory since registers can't be addressed.
* Thread-local scope
* Cached in L1 and L2 cache, so register spilling may cause less slowdown on newer cards.

#### Shared Memory

Very fast (register speeds).

* Shared between threads within each block (block-scope).
* Successive dwords reside in different banks.
* 16 banks in compute capability 1.0, 32 in compute capability 2.0
* Fastest when all threads read either entirely different banks or the same value.
* Used to enable fast communication between threads in a block.
* Block-local scope.

Created shared memory with the `__shared__` keyword. Arrays must be sized at compile time, no dynamic allocation.

```cuda
__global__ void foo(int* a, int* b) {
  __shared__ int smem[1024];
}
```

Or, use `extern` with `__shared__` to declare a dynamic array whose size can be
dynamically determined by kernel launch configuration.

```cuda
__global__ void foo(int* a, int* b) {
  extern __shared__ int smem[];
}

int main() {
  int a = 3, b = 2;
  int *d_a, *d_b;

  cudaMalloc(&d_a, sizeof(int));
  cudaMalloc(&d_b, sizeof(int));

  cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

  // Third parameter is bytes to allocate per block
  foo<<<1, 1, 1024>>>

  cudaFree(d_a);
  cudaFree(d_b);

  return 0;
}
```

#### Caches

On compute 2.0 and above, each multiprocessor has an L1 cache, and an L2 cache is shared between all multiprocessors.
These are used by global and local memory.

* The L1 cache is very fast, and can be used as shared memory.
* L1 uses same bytes as shared memory.
* L1 cache can be configured.
  * Use as 48k shared memory with 16k L1 or 16k shared with 48k L1.
* All global memory accesses go through the L2 cache, including those by the CPU.
* You can turn off caching with a compiler option.

#### Constant Memory

* Part of the GPU's main memory
* Has its own per-multiprocessor cache
* Read-only in the GPU, R/W on the CPU
* Very fast if all threads read the same address
* Shared between all threads

#### Texture Memory

Also part of device memory like global, local and constant memory.

* Has extra addressing tricks because it's designed for indexing / sampling a 2D image
* Can interpolate values
* Lower bandwidth than global memory's L1 cache
* Shared between all threads
* Per-thread caches prone to data races if threads mutate texture data
  * NVIDIA says this is undefined behavior

#### Registers

Fastest memory access, used automatically by the compiler. Basically CPU registers.

* GPUs, unlike CPUs, have thousands of registers.
* Registers are thread-local.
* Still, using fewer registers per thread can result in better performance.

### Synchronization

`__syncthreads()` is a function that acts as a barrier, and waits for all threads in the block to reach the callsite.

```cuda
__shared__ int i;
i = 0;
__syncthreads(); // without syncthreads we could end up with 0, 1, or 2.
if (threadIdx.x == 0) i++;
__syncthreads();
if (threadIdx.x == 1) i++;
```

## API

* Kernels always return void.
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
