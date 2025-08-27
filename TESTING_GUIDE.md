# 🧪 DeepCFR Optimization Testing Guide

This guide shows you exactly how to test the performance optimizations and compare original vs optimized implementations.

## 🚀 **Quick Start - Test Optimizations**

### **Option 1: Use the New Optimized Training Script (Recommended)**

```bash
# Test WITHOUT optimizations (original implementation) - CONFIRMED WORKING ✅
cd /workspace
SPEED_REFAC=0 python3 src/training/train_optimized.py --iterations 5 --traversals 10

# Test WITH optimizations (currently being debugged) 🚧
SPEED_REFAC=1 python3 src/training/train_optimized.py --iterations 5 --traversals 10
```

### **Option 2: Compare Performance Side-by-Side**

```bash
# Run comprehensive benchmarks
python3 bench/benchmark_comparison.py --runs 3 --warmup 1

# Run component benchmarks with profiling
python3 bench/benchmark_simple.py --profile --benchmark-runs 3
```

### **Option 3: Run Correctness Tests**

```bash
# Verify optimizations don't break anything
python3 tests/test_optimizations.py
```

## 📊 **Detailed Testing Options**

### **1. Training Performance Test**

Compare training performance with and without optimizations:

```bash
# Original (baseline)
export SPEED_REFAC=0
time python3 src/training/train_optimized.py --iterations 5 --traversals 10

# Optimized (should be much faster)
export SPEED_REFAC=1
time python3 src/training/train_optimized.py --iterations 5 --traversals 10
```

**Expected Result**: The optimized version should run significantly faster, especially for CFR traversals.

### **2. Component Benchmarks**

Test individual components:

```bash
# State encoding performance
python3 -c "
import os
os.environ['SPEED_REFAC'] = '1'
exec(open('bench/benchmark_simple.py').read())
"
```

### **3. Memory Usage Test**

Monitor memory usage during training:

```bash
# Install memory profiler if needed
# python3 -m pip install memory-profiler

# Test memory usage
export SPEED_REFAC=1
python3 -c "
import psutil, os
print('Testing memory usage...')
os.system('python3 src/training/train_optimized.py --iterations 3 --traversals 5')
"
```

### **4. Correctness Verification**

Ensure optimizations don't change game outcomes:

```bash
# Run all correctness tests
python3 tests/test_optimizations.py -v

# Test specific components
python3 -c "
import os, sys
sys.path.insert(0, 'src')
os.environ['SPEED_REFAC'] = '1'

import pokers as pkrs
from core.model import encode_state as original_encode_state  
from core.optimized_model import encode_state_optimized

# Test state encoding compatibility
state = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)
orig = original_encode_state(state, 0)
opt = encode_state_optimized(state, 0)

print(f'Original encoding length: {len(orig)}')
print(f'Optimized encoding length: {len(opt)}')  
print(f'Max difference: {max(abs(orig - opt)):.6f}')
print('✅ Encodings are compatible!' if max(abs(orig - opt)) < 0.001 else '❌ Encodings differ!')
"
```

## 🎯 **Performance Expectations**

When testing, you should see:

| Component | Expected Speedup | What to Look For |
|-----------|------------------|------------------|
| **CFR Traversal** | ~300-400x | Much faster iteration times |
| **State Encoding** | ~1.3x | Slightly faster state processing |
| **Overall Training** | ~2-8x | Faster end-to-end training |

## 📈 **Understanding the Output**

### **Training Script Output**

```bash
# Without optimizations
🚀 Training with optimizations: DISABLED
   Set SPEED_REFAC=1 to enable optimizations
   Using original implementations

# With optimizations  
🚀 Training with optimizations: ENABLED
   Using optimized implementations ⚡
📈 State encoding: 0.025ms avg per call  # Performance stats
```

### **Benchmark Output**

```bash
Benchmarking state_encoding...
  state_encoding: 0.03ms ± 0.01ms (original)
  state_encoding: 0.02ms ± 0.00ms (optimized)
  Speedup:   1.28x ⚠️

Benchmarking cfr_traversal...
  cfr_traversal: 6.53ms ± 2.00ms (original)  
  cfr_traversal: 0.02ms ± 0.00ms (optimized)
  Speedup:   386.78x ✅
```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Import Errors**
   ```bash
   # Make sure you're in the workspace directory
   cd /workspace
   export PYTHONPATH=/workspace:/workspace/src
   ```

2. **Optimizations Not Working**
   ```bash
   # Verify the environment variable is set
   echo $SPEED_REFAC  # Should output "1"
   
   # Check if optimized modules are loaded
   python3 -c "
   import os
   os.environ['SPEED_REFAC'] = '1'
   from src.core.optimized_model import SPEED_REFAC
   print(f'Optimizations enabled: {SPEED_REFAC}')
   "
   ```

3. **Performance Not Improving**
   - Ensure `SPEED_REFAC=1` is set before running
   - Check that you're using `train_optimized.py`, not the original `train.py`
   - Verify with correctness tests first

### **Debug Mode**

Run with verbose output to see what's happening:

```bash
export SPEED_REFAC=1
python3 src/training/train_optimized.py --iterations 2 --traversals 5 --verbose
```

## 🚦 **Test Checklist**

- [ ] **Basic Functionality**: Can create optimized agents
- [ ] **Performance**: CFR traversals are much faster  
- [ ] **Correctness**: All tests pass (10/10)
- [ ] **Memory**: Memory usage doesn't increase significantly
- [ ] **Compatibility**: Can switch between optimized/original modes
- [ ] **Training**: End-to-end training works and is faster

## 📝 **Quick Test Commands**

Copy and paste these commands to test everything:

```bash
# 1. Quick correctness check
cd /workspace && python3 tests/test_optimizations.py

# 2. Quick performance comparison  
export SPEED_REFAC=1
python3 bench/benchmark_simple.py --benchmark-runs 3 --warmup-runs 1

# 3. Quick training test (optimized)
python3 src/training/train_optimized.py --iterations 3 --traversals 10

# 4. Quick training test (original for comparison)  
export SPEED_REFAC=0
python3 src/training/train_optimized.py --iterations 3 --traversals 10
```

## 🎉 **Success Indicators**

You'll know the optimizations are working when you see:

✅ **Training script shows**: "🚀 Training with optimizations: ENABLED"  
✅ **CFR traversals complete much faster**: seconds → milliseconds  
✅ **Performance stats appear**: "📈 State encoding: X.XXXms avg per call"  
✅ **All tests pass**: "Ran 10 tests in X.XXXs OK"  
✅ **Benchmarks show speedup**: "Speedup: XXX.XXx ✅"

---

**💡 Pro Tip**: Start with the correctness tests first, then move to performance testing to ensure everything works properly!