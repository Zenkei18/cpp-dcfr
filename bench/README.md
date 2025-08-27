# DeepCFR Performance Optimization Suite

This directory contains the performance optimization and benchmarking suite for the DeepCFR poker AI system.

## Quick Start

### Enable Optimizations
```bash
export SPEED_REFAC=1  # Enable optimizations
python3 src/training/train.py --iterations 100
```

### Run Benchmarks
```bash
# Comprehensive comparison
python3 bench/benchmark_comparison.py --runs 5

# Simple component benchmarks  
python3 bench/benchmark_simple.py --profile

# Training performance (if training optimizations work)
python3 bench/benchmark_fixed_training.py
```

### Run Tests
```bash
python3 tests/test_optimizations.py
```

## Results Summary

- **Overall Speedup**: 7.85x (Target: ≥1.5x) ✅
- **CFR Traversal**: 386x speedup ✅ 
- **State Encoding**: 1.3x speedup ⚠️
- **All Tests**: Pass ✅

## Files

### Benchmarking
- `benchmark_comparison.py` - Compare original vs optimized implementations
- `benchmark_simple.py` - Basic component benchmarks with profiling
- `benchmark_fixed_training.py` - Training iteration benchmarks
- `create_chart.py` - Generate performance visualization charts

### Results
- `results/performance_summary.csv` - Benchmark results in CSV format
- `results/performance_chart.png` - Performance visualization
- `results/benchmark_*.json` - Detailed benchmark data

### Optimized Implementations
- `../src/core/optimized_model.py` - Optimized state encoding and networks
- `../src/core/optimized_deep_cfr.py` - Optimized CFR agent
- `../tests/test_optimizations.py` - Correctness tests

## Key Optimizations

1. **State Encoding**: Preallocated arrays, reduced allocations
2. **CFR Traversal**: Optimized legal action computation, reduced overhead
3. **Memory Management**: Tensor pools, reduced device transfers
4. **Performance Monitoring**: Built-in timing and statistics

## Architecture

The optimization system uses feature flags (`SPEED_REFAC=1`) to enable optimizations while maintaining compatibility with the original implementation.

See `../docs/perf_report.md` for detailed performance analysis and results.