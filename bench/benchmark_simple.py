#!/usr/bin/env python3
"""
Simplified benchmarking suite for DeepCFR poker training.
This version avoids complex imports to focus on core performance bottlenecks.
"""

import os
import sys
import time
import json
import statistics
import numpy as np
import torch
import cProfile
import pstats
import io
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import only what we need for benchmarking
import pokers as pkrs


@dataclass
class BenchmarkResult:
    """Container for benchmark timing results."""
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    runs: int
    total_time: float
    memory_mb: Optional[float] = None
    extra_info: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.extra_info is None:
            result['extra_info'] = {}
        return result


class SimpleBenchmarker:
    """Simplified benchmarker focusing on core bottlenecks."""
    
    def __init__(self, device='auto', results_dir='bench/results', warmup_runs=3, benchmark_runs=10):
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.results_dir = results_dir
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = {}
        
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Simplified Performance Benchmarker initialized")
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Warmup runs: {warmup_runs}, Benchmark runs: {benchmark_runs}")
        print("-" * 60)

    @contextmanager
    def timer(self):
        """Context manager for precise timing."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            self.last_time = end - start

    def get_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if self.device == 'cuda' and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return None

    def benchmark_function(self, func: Callable, name: str, 
                          setup_func: Optional[Callable] = None, 
                          **kwargs) -> BenchmarkResult:
        """Benchmark a function with multiple runs."""
        times = []
        memory_usage = []
        
        print(f"Benchmarking {name}...")
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                if setup_func:
                    args = setup_func()
                    if args:
                        kwargs.update(args)
                with self.timer():
                    result = func(**kwargs)
            except Exception as e:
                print(f"Warning: Error in warmup for {name}: {e}")
                continue
                
        # Benchmark runs
        for i in range(self.benchmark_runs):
            try:
                if setup_func:
                    args = setup_func()
                    if args:
                        kwargs.update(args)
                
                mem_before = self.get_memory_usage()
                with self.timer():
                    result = func(**kwargs)
                mem_after = self.get_memory_usage()
                
                times.append(self.last_time)
                if mem_before is not None and mem_after is not None:
                    memory_usage.append(mem_after - mem_before)
                    
            except Exception as e:
                print(f"Warning: Error in run {i+1} for {name}: {e}")
                continue
                
        if not times:
            print(f"Error: No successful runs for {name}")
            return BenchmarkResult(name, 0, 0, 0, 0, 0, 0)
            
        # Calculate statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)
        
        avg_memory = statistics.mean(memory_usage) if memory_usage else None
        
        result = BenchmarkResult(
            name=name,
            mean_time=mean_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            runs=len(times),
            total_time=total_time,
            memory_mb=avg_memory
        )
        
        print(f"  {name}: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms "
              f"(min: {min_time*1000:.2f}ms, max: {max_time*1000:.2f}ms)")
        if avg_memory is not None:
            print(f"  Memory: {avg_memory:.2f} MB")
            
        self.results[name] = result
        return result

    def benchmark_state_encoding(self):
        """Benchmark state encoding performance."""
        # Import here to avoid cascade issues
        from src.core.model import encode_state
        
        def setup_state():
            state = pkrs.State.from_seed(
                n_players=6,
                button=0,
                sb=1,
                bb=2,
                stake=200.0,
                seed=42
            )
            return {'state': state, 'player_id': 0}
            
        return self.benchmark_function(
            func=encode_state,
            name="state_encoding",
            setup_func=setup_state
        )

    def benchmark_tensor_operations(self):
        """Benchmark basic tensor operations."""
        def create_tensor():
            return torch.randn(500, device=self.device)
            
        def tensor_ops(tensor):
            # Simulate typical operations in the CFR algorithm
            x = tensor.unsqueeze(0)
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            x = x.cpu().numpy()
            return x
            
        tensor = create_tensor()
        return self.benchmark_function(
            func=tensor_ops,
            name="tensor_operations",
            tensor=tensor
        )

    def benchmark_device_transfers(self):
        """Benchmark device transfer operations."""
        def setup_transfer():
            data = np.random.randn(500).astype(np.float32)
            cpu_tensor = torch.from_numpy(data)
            
            if self.device == 'cuda':
                gpu_tensor = cpu_tensor.cuda()
                return {'cpu_tensor': cpu_tensor, 'gpu_tensor': gpu_tensor}
            else:
                return {'cpu_tensor': cpu_tensor, 'gpu_tensor': cpu_tensor}
                
        def cpu_to_device(cpu_tensor, **kwargs):
            return cpu_tensor.to(self.device)
            
        def device_to_numpy(gpu_tensor, **kwargs):
            return gpu_tensor.cpu().numpy()
            
        # Benchmark transfers
        result1 = self.benchmark_function(
            func=cpu_to_device,
            name="cpu_to_device_transfer",
            setup_func=setup_transfer
        )
        
        result2 = self.benchmark_function(
            func=device_to_numpy,
            name="device_to_numpy_transfer",
            setup_func=setup_transfer
        )
        
        return result1, result2

    def benchmark_numpy_operations(self):
        """Benchmark NumPy operations commonly used in the code."""
        def setup_numpy():
            return {'data': np.random.randn(500)}
            
        def numpy_ops(data):
            # Simulate state encoding operations
            encoded = []
            
            # Zeros arrays (common in encode_state)
            hand_enc = np.zeros(52)
            community_enc = np.zeros(52)
            stage_enc = np.zeros(5)
            
            # Array operations
            hand_enc[np.random.randint(0, 52, 2)] = 1
            community_enc[np.random.randint(0, 52, 5)] = 1
            stage_enc[np.random.randint(0, 5)] = 1
            
            # Concatenation (major operation in encode_state)
            result = np.concatenate([hand_enc, community_enc, stage_enc, data[:100]])
            return result
            
        return self.benchmark_function(
            func=numpy_ops,
            name="numpy_operations",
            setup_func=setup_numpy
        )

    def benchmark_poker_state_creation(self):
        """Benchmark creating poker states."""
        def create_state():
            return pkrs.State.from_seed(
                n_players=6,
                button=np.random.randint(0, 6),
                sb=1,
                bb=2,
                stake=200.0,
                seed=np.random.randint(0, 10000)
            )
            
        return self.benchmark_function(
            func=create_state,
            name="poker_state_creation"
        )

    def run_basic_benchmarks(self):
        """Run basic performance benchmarks."""
        print(f"\n{'='*60}")
        print("BASIC DEEPCFR PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
        
        results_summary = {
            'device': self.device,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmarks': {}
        }
        
        # Run benchmarks
        benchmarks = [
            ('poker_state_creation', self.benchmark_poker_state_creation),
            ('state_encoding', self.benchmark_state_encoding),
            ('numpy_operations', self.benchmark_numpy_operations),
            ('tensor_operations', self.benchmark_tensor_operations),
            ('device_transfers', self.benchmark_device_transfers)
        ]
        
        for name, benchmark_func in benchmarks:
            try:
                print(f"\n{'-'*40}")
                print(f"Running {name}")
                print(f"{'-'*40}")
                
                result = benchmark_func()
                if isinstance(result, tuple):  # Multiple results
                    for r in result:
                        results_summary['benchmarks'][r.name] = r.to_dict()
                else:
                    results_summary['benchmarks'][result.name] = result.to_dict()
                    
            except Exception as e:
                print(f"Error in {name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        timestamp = int(time.time())
        results_file = os.path.join(self.results_dir, f'benchmark_simple_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Results saved to: {results_file}")
        print(f"Total benchmarks run: {len(results_summary['benchmarks'])}")
        
        print(f"\nPerformance Results:")
        print("-" * 40)
        for name, result in results_summary['benchmarks'].items():
            print(f"{name:25s}: {result['mean_time']*1000:8.2f}ms ± {result['std_time']*1000:.2f}ms")
        
        return results_summary

    def profile_state_encoding(self):
        """Profile state encoding to find bottlenecks."""
        from src.core.model import encode_state
        
        print(f"\nProfiling state encoding...")
        
        # Setup
        states = []
        for i in range(50):  # Create multiple states for profiling
            state = pkrs.State.from_seed(
                n_players=6,
                button=i % 6,
                sb=1,
                bb=2,
                stake=200.0,
                seed=i + 100
            )
            states.append(state)
        
        def encode_multiple_states():
            for state in states:
                encode_state(state, 0)
        
        # Profile the state encoding
        profiler = cProfile.Profile()
        profiler.enable()
        
        encode_multiple_states()
        
        profiler.disable()
        
        # Analyze results
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        profile_output = stream.getvalue()
        
        # Save detailed profile
        profile_file = os.path.join(self.results_dir, f'profile_state_encoding_{int(time.time())}.txt')
        with open(profile_file, 'w') as f:
            f.write(profile_output)
        
        print(f"Detailed profile saved to {profile_file}")
        
        # Extract and return top bottlenecks
        lines = profile_output.split('\n')
        bottlenecks = []
        
        stats_start = -1
        for i, line in enumerate(lines):
            if 'ncalls' in line and 'tottime' in line:
                stats_start = i + 1
                break
        
        if stats_start > 0:
            for i in range(stats_start, min(stats_start + 10, len(lines))):
                line = lines[i].strip()
                if line and not line.startswith('-'):
                    bottlenecks.append(line)
        
        print("\nTop State Encoding Bottlenecks:")
        print("-" * 80)
        for i, bottleneck in enumerate(bottlenecks[:5], 1):
            print(f"{i}. {bottleneck}")
        
        return bottlenecks[:5]


def main():
    """Main benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified DeepCFR Performance Benchmarker')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to run benchmarks on')
    parser.add_argument('--warmup-runs', type=int, default=2,
                       help='Number of warmup runs')
    parser.add_argument('--benchmark-runs', type=int, default=5,
                       help='Number of benchmark runs to average')
    parser.add_argument('--results-dir', default='bench/results',
                       help='Directory to save results')
    parser.add_argument('--profile', action='store_true',
                       help='Include profiling step')
    
    args = parser.parse_args()
    
    benchmarker = SimpleBenchmarker(
        device=args.device,
        results_dir=args.results_dir,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs
    )
    
    # Run basic benchmarks
    results = benchmarker.run_basic_benchmarks()
    
    # Run profiling if requested
    if args.profile:
        try:
            bottlenecks = benchmarker.profile_state_encoding()
        except Exception as e:
            print(f"Profiling failed: {e}")
    
    return results


if __name__ == '__main__':
    main()