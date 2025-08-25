# Performance Guide: DeepCFR C++ Implementation

This guide provides performance optimization tips and best practices for the DeepCFR C++ implementation.

## Performance Overview

The C++ implementation delivers substantial performance improvements over the Python version:

| Component | Performance Gain | Notes |
|-----------|------------------|-------|
| Core Algorithm | 15-20x | Faster CFR traversals, action selection, and state transitions |
| Neural Network | 10-15x | More efficient forward and backward passes |
| Memory Usage | 3-5x reduction | More efficient memory layout and management |
| Training Speed | 10-30x | Faster iterations lead to quicker convergence |

## Bottlenecks and Optimizations

### 1. State Encoding

State encoding is one of the most frequently called functions during training and inference. The C++ implementation optimizes this by:

- Using contiguous memory layouts
- Avoiding temporary allocations
- Leveraging SIMD instructions where possible

**Tip**: Pre-allocate tensors when processing batches of states to avoid repeated allocations.

### 2. Memory Management

Memory management is critical for large-scale training. The C++ implementation provides:

- Custom memory pools for frequently allocated objects
- Efficient tensor management to reduce fragmentation
- Lower overhead for experience replay buffer operations

**Tip**: Adjust the `capacity` parameter of `PrioritizedMemory` based on your hardware capabilities.

### 3. Neural Network Operations

Neural network operations are optimized through:

- Direct integration with libtorch's C++ API
- Batch processing of states
- GPU acceleration (when available)

**Tip**: Use batch sizes that are powers of 2 (e.g., 32, 64, 128, 256) for optimal performance.

## Hardware Considerations

### CPU Performance

The C++ implementation can efficiently utilize multiple CPU cores:

```cpp
// Set thread count for parallelism
torch::set_num_threads(8);  // Adjust based on your CPU
```

**Recommended**: Modern CPUs with AVX2 or AVX-512 support will see the best performance due to vectorization.

### GPU Acceleration

When using CUDA-enabled GPUs:

```cpp
// Create agent on GPU
DeepCFRAgent agent(0, 6, 4, "cuda");
```

**Note**: For small models, the CPU may outperform the GPU due to data transfer overhead. Benchmark both options.

### Memory Requirements

- **RAM**: Minimum 8GB, recommended 16GB+ for large training runs
- **VRAM** (if using GPU): Minimum 4GB, recommended 8GB+

## Profiling and Benchmarking

The C++ implementation includes benchmarking tools:

```bash
# Run all benchmarks
./benchmarks/deepcfr_benchmarks

# Run specific benchmark
./benchmarks/deepcfr_benchmarks --benchmark_filter=BM_StateEncoding
```

For detailed profiling:

1. CPU profiling with `perf`:
   ```bash
   perf record -g ./benchmarks/deepcfr_benchmarks
   perf report
   ```

2. Memory profiling with `valgrind`:
   ```bash
   valgrind --tool=massif ./benchmarks/deepcfr_benchmarks
   ms_print massif.out.xxxx
   ```

## Configuration Options

### Training Configuration

Adjust these parameters for optimal performance:

```cpp
// Larger batch sizes generally improve throughput
const size_t batch_size = 512;

// Higher memory capacity improves learning but increases memory usage
PrioritizedMemory memory(10000000);  // 10M experiences

// Number of traversals affects data collection speed
const int traversals_per_iteration = 1000;
```

### System Configuration

1. **Thread affinity**: On multi-socket systems, setting thread affinity can improve performance:
   ```bash
   numactl --cpunodebind=0 --membind=0 ./deepcfr-train
   ```

2. **Memory policy**: For large training runs:
   ```bash
   echo 1 > /proc/sys/vm/overcommit_memory  # Allow memory overcommit
   ```

## Common Pitfalls

1. **Tensor copies**: Avoid unnecessary tensor copies, use references or views where possible.

2. **Python GIL**: When using Python bindings, be aware that the Global Interpreter Lock can limit parallelism.

3. **Memory leaks**: Use monitoring tools like `valgrind` to detect and fix memory leaks.

## Advanced Optimizations

For users seeking maximum performance:

1. **Custom CUDA kernels**: Implement specialized CUDA kernels for critical operations.

2. **Quantization**: Use 16-bit or 8-bit quantization for models (requires custom implementation).

3. **Distributed training**: Implement distributed training across multiple machines.

## Benchmarking Your Setup

Run the included benchmarks to compare your setup against reference numbers:

```bash
cd build
./bin/deepcfr_benchmarks
```

Expected results on reference hardware (AMD Ryzen 9 5950X):

- State creation: ~1,000,000 states/sec
- State encoding: ~500,000 states/sec
- NN forward pass (batch=128): ~10,000 batches/sec
- CFR traversals: ~1,000 traversals/sec
