# DeepCFR C++ Implementation

This is a C++ port of the DeepCFR (Deep Counterfactual Regret Minimization) Poker AI, originally implemented in Python. The C++ implementation aims to preserve behavior and public APIs while substantially improving runtime performance and memory efficiency.

## Features

- High-performance C++20 implementation
- Python bindings via pybind11
- Identical APIs to the Python version
- Comprehensive test suite and benchmarks
- CMake-based build system

## Requirements

- C++20 compatible compiler:
  - GCC >= 13
  - Clang >= 16
  - MSVC >= 14.3 (Visual Studio 2022)
- CMake >= 3.16
- PyTorch C++ API (LibTorch)
- spdlog

## Building from Source

### Linux/macOS

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Windows

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Running Tests

```bash
cd build
ctest --output-on-failure
```

## Running Benchmarks

```bash
cd build
./bin/deepcfr_benchmarks
```

## Python Bindings

The C++ implementation can be used from Python via the provided bindings:

```bash
cd bindings
pip install -e .
```

Then in Python:

```python
from deepcfr_cpp import DeepCFRAgent, encode_state

# Create an agent
agent = DeepCFRAgent(player_id=0, num_players=6)

# Train and use it just like the Python version
```

## License

MIT License - See LICENSE file for details.
