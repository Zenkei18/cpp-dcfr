#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for DeepCFR poker training.

This module provides both micro-benchmarks (individual functions) and macro-benchmarks 
(end-to-end training steps) to identify performance bottlenecks and measure speedup improvements.
"""

import os
import sys
import time
import json
import traceback
import statistics
import numpy as np
import torch
import cProfile
import pstats
import io
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pokers as pkrs
from core.deep_cfr import DeepCFRAgent
from core.model import encode_state, PokerNetwork, set_verbose
from agents.random_agent import RandomAgent
from utils.settings import set_strict_checking


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


class PerformanceBenchmarker:
    """Main benchmarking class with micro and macro benchmarks."""
    
    def __init__(self, device='auto', results_dir='bench/results', warmup_runs=3, benchmark_runs=10):
        """
        Initialize the benchmarker.
        
        Args:
            device: Device to run benchmarks on ('auto', 'cpu', 'cuda')
            results_dir: Directory to save benchmark results
            warmup_runs: Number of warmup runs before measuring
            benchmark_runs: Number of benchmark runs to average
        """
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
        
        # Setup verbose and error handling
        set_verbose(False)  # Reduce noise during benchmarking
        set_strict_checking(False)  # Continue on errors for benchmarking
        
        print(f"Performance Benchmarker initialized")
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
        """
        Benchmark a function with multiple runs.
        
        Args:
            func: Function to benchmark
            name: Name for the benchmark
            setup_func: Optional setup function called before each run
            **kwargs: Additional arguments passed to func
        """
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

    # ========================================
    # MICRO-BENCHMARKS
    # ========================================
    
    def benchmark_state_encoding(self):
        """Benchmark state encoding performance."""
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

    def benchmark_network_forward(self):
        """Benchmark neural network forward pass."""
        def setup_network():
            # Create a sample state and encode it
            state = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)
            encoded = encode_state(state, 0)
            
            # Create network and move to device
            input_size = len(encoded)
            network = PokerNetwork(input_size=input_size, hidden_size=256, num_actions=3).to(self.device)
            
            # Create input tensor
            input_tensor = torch.FloatTensor(encoded).unsqueeze(0).to(self.device)
            
            return {'network': network, 'input_tensor': input_tensor}
            
        def forward_pass(network, input_tensor):
            with torch.no_grad():
                return network(input_tensor)
                
        return self.benchmark_function(
            func=forward_pass,
            name="network_forward_pass",
            setup_func=setup_network
        )

    def benchmark_tensor_device_transfer(self):
        """Benchmark tensor device transfers."""
        def setup_transfer():
            # Create a typical state encoding tensor
            data = np.random.randn(500).astype(np.float32)
            cpu_tensor = torch.from_numpy(data)
            
            if self.device == 'cuda':
                gpu_tensor = cpu_tensor.cuda()
                return {'cpu_tensor': cpu_tensor, 'gpu_tensor': gpu_tensor}
            else:
                return {'cpu_tensor': cpu_tensor, 'gpu_tensor': cpu_tensor}
                
        def transfer_to_device(cpu_tensor, **kwargs):
            return cpu_tensor.to(self.device)
            
        def transfer_to_numpy(gpu_tensor, **kwargs):
            return gpu_tensor.cpu().numpy()
            
        # Benchmark CPU -> Device transfer
        result1 = self.benchmark_function(
            func=transfer_to_device,
            name="cpu_to_device_transfer",
            setup_func=setup_transfer
        )
        
        # Benchmark Device -> CPU -> NumPy transfer  
        result2 = self.benchmark_function(
            func=transfer_to_numpy,
            name="device_to_numpy_transfer",
            setup_func=setup_transfer
        )
        
        return result1, result2

    def benchmark_memory_operations(self):
        """Benchmark memory operations like sampling from prioritized replay."""
        def setup_memory():
            from core.deep_cfr import PrioritizedMemory
            
            # Create and populate a memory buffer
            memory = PrioritizedMemory(capacity=10000, alpha=0.6)
            
            # Add sample experiences
            for i in range(1000):
                experience = (
                    np.random.randn(500),  # state
                    np.random.randn(20),   # opponent features
                    np.random.randint(0, 3),  # action
                    np.random.rand(),      # bet size
                    np.random.randn()      # regret
                )
                memory.add(experience, priority=abs(np.random.randn()) + 0.1)
                
            return {'memory': memory, 'batch_size': 128, 'beta': 0.4}
            
        def sample_batch(memory, batch_size, beta):
            return memory.sample(batch_size, beta)
            
        return self.benchmark_function(
            func=sample_batch,
            name="prioritized_memory_sampling",
            setup_func=setup_memory
        )

    # ========================================
    # CFR-SPECIFIC BENCHMARKS  
    # ========================================
    
    def benchmark_cfr_traverse_single(self):
        """Benchmark a single CFR traversal."""
        def setup_cfr():
            # Create agent
            agent = DeepCFRAgent(player_id=0, num_players=6, device=self.device)
            
            # Create initial state
            state = pkrs.State.from_seed(
                n_players=6,
                button=0,
                sb=1,
                bb=2,
                stake=200.0,
                seed=42
            )
            
            # Create random opponents
            random_agents = [RandomAgent(i) for i in range(6)]
            
            return {'agent': agent, 'state': state, 'iteration': 1, 'random_agents': random_agents}
            
        def single_traverse(agent, state, iteration, random_agents):
            return agent.cfr_traverse(state, iteration, random_agents)
            
        return self.benchmark_function(
            func=single_traverse,
            name="cfr_single_traversal",
            setup_func=setup_cfr
        )

    def benchmark_advantage_training(self):
        """Benchmark advantage network training."""
        def setup_training():
            agent = DeepCFRAgent(player_id=0, num_players=6, device=self.device)
            
            # Populate advantage memory with some data
            for i in range(500):
                experience = (
                    np.random.randn(500),  # state  
                    np.random.randn(20),   # opponent features
                    np.random.randint(0, 3),  # action
                    np.random.rand(),      # bet size
                    np.random.randn()      # regret
                )
                agent.advantage_memory.add(experience, priority=abs(np.random.randn()) + 0.1)
                
            return {'agent': agent, 'batch_size': 128, 'epochs': 1}
            
        def train_advantage(agent, batch_size, epochs):
            return agent.train_advantage_network(batch_size=batch_size, epochs=epochs)
            
        return self.benchmark_function(
            func=train_advantage,
            name="advantage_network_training",
            setup_func=setup_training
        )

    def benchmark_strategy_training(self):
        """Benchmark strategy network training."""
        def setup_strategy():
            agent = DeepCFRAgent(player_id=0, num_players=6, device=self.device)
            
            # Populate strategy memory
            for i in range(500):
                strategy = np.zeros(3)
                strategy[np.random.randint(0, 3)] = 1.0  # One-hot strategy
                
                experience = (
                    np.random.randn(500),  # state
                    np.random.randn(20),   # opponent features  
                    strategy,              # strategy
                    np.random.rand(),      # bet size
                    i + 1                  # iteration
                )
                agent.strategy_memory.append(experience)
                
            return {'agent': agent, 'batch_size': 128, 'epochs': 1}
            
        def train_strategy(agent, batch_size, epochs):
            return agent.train_strategy_network(batch_size=batch_size, epochs=epochs)
            
        return self.benchmark_function(
            func=train_strategy,
            name="strategy_network_training", 
            setup_func=setup_strategy
        )

    # ========================================
    # MACRO-BENCHMARKS
    # ========================================
    
    def benchmark_training_iteration(self, size='S'):
        """
        Benchmark a complete training iteration.
        
        Args:
            size: Benchmark size ('S' for small, 'M' for medium, 'L' for large)
        """
        size_configs = {
            'S': {'traversals': 10, 'memory_size': 1000, 'batch_size': 32},
            'M': {'traversals': 50, 'memory_size': 5000, 'batch_size': 128},
            'L': {'traversals': 200, 'memory_size': 20000, 'batch_size': 256}
        }
        
        config = size_configs[size]
        
        def setup_full_iteration():
            # Create agent with specified memory size
            agent = DeepCFRAgent(
                player_id=0, 
                num_players=6, 
                memory_size=config['memory_size'],
                device=self.device
            )
            
            # Create random agents
            random_agents = [RandomAgent(i) for i in range(6)]
            
            return {
                'agent': agent,
                'random_agents': random_agents,
                'traversals': config['traversals'],
                'batch_size': config['batch_size']
            }
            
        def full_training_iteration(agent, random_agents, traversals, batch_size):
            iteration = 1
            
            # Run traversals to collect data
            for t in range(traversals):
                state = pkrs.State.from_seed(
                    n_players=6,
                    button=t % 6,
                    sb=1,
                    bb=2, 
                    stake=200.0,
                    seed=t + 1000
                )
                agent.cfr_traverse(state, iteration, random_agents)
                
            # Train networks
            adv_loss = agent.train_advantage_network(batch_size=batch_size, epochs=1)
            strat_loss = agent.train_strategy_network(batch_size=batch_size, epochs=1)
            
            return adv_loss, strat_loss
            
        return self.benchmark_function(
            func=full_training_iteration,
            name=f"full_training_iteration_size_{size}",
            setup_func=setup_full_iteration
        )

    # ========================================
    # PROFILING
    # ========================================
    
    def profile_training_iteration(self, size='M', profile_file=None):
        """
        Profile a training iteration and return top bottlenecks.
        
        Args:
            size: Size of the benchmark ('S', 'M', 'L')
            profile_file: Optional file to save detailed profile results
        """
        print(f"\nProfiling training iteration (size {size})...")
        
        size_configs = {
            'S': {'traversals': 10, 'memory_size': 1000},
            'M': {'traversals': 50, 'memory_size': 5000}, 
            'L': {'traversals': 200, 'memory_size': 20000}
        }
        
        config = size_configs[size]
        
        # Setup
        agent = DeepCFRAgent(
            player_id=0,
            num_players=6,
            memory_size=config['memory_size'],
            device=self.device
        )
        
        random_agents = [RandomAgent(i) for i in range(6)]
        
        def training_iteration():
            iteration = 1
            
            # Run traversals
            for t in range(config['traversals']):
                state = pkrs.State.from_seed(
                    n_players=6,
                    button=t % 6,
                    sb=1,
                    bb=2,
                    stake=200.0,
                    seed=t + 2000
                )
                agent.cfr_traverse(state, iteration, random_agents)
                
            # Train networks  
            agent.train_advantage_network(batch_size=128, epochs=1)
            agent.train_strategy_network(batch_size=128, epochs=1)
        
        # Profile the training iteration
        profiler = cProfile.Profile()
        profiler.enable()
        
        training_iteration()
        
        profiler.disable()
        
        # Analyze results
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(50)  # Top 50 functions
        
        profile_output = stream.getvalue()
        
        # Save to file if requested
        if profile_file:
            profile_path = os.path.join(self.results_dir, profile_file)
            with open(profile_path, 'w') as f:
                f.write(profile_output)
            print(f"Detailed profile saved to {profile_path}")
        
        # Extract top bottlenecks
        lines = profile_output.split('\n')
        bottlenecks = []
        
        # Find the stats table
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
        
        print("\nTop 5 Performance Bottlenecks:")
        print("-" * 80)
        for i, bottleneck in enumerate(bottlenecks[:5], 1):
            print(f"{i}. {bottleneck}")
        
        return bottlenecks[:5]

    # ========================================
    # MAIN BENCHMARK RUNNER
    # ========================================
    
    def run_all_benchmarks(self, include_profiling=True, sizes=['S', 'M']):
        """Run all benchmarks and save results."""
        print(f"\n{'='*60}")
        print("STARTING COMPREHENSIVE DEEPCFR PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
        
        results_summary = {
            'device': self.device,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'micro_benchmarks': {},
            'macro_benchmarks': {},
            'bottlenecks': []
        }
        
        # Run micro-benchmarks
        print("\n" + "="*40)
        print("MICRO-BENCHMARKS")
        print("="*40)
        
        micro_benchmarks = [
            ('state_encoding', self.benchmark_state_encoding),
            ('network_forward', self.benchmark_network_forward),
            ('memory_operations', self.benchmark_memory_operations),
            ('cfr_traverse_single', self.benchmark_cfr_traverse_single),
            ('advantage_training', self.benchmark_advantage_training),
            ('strategy_training', self.benchmark_strategy_training)
        ]
        
        for name, benchmark_func in micro_benchmarks:
            try:
                result = benchmark_func()
                if isinstance(result, tuple):  # Handle functions that return multiple results
                    for r in result:
                        results_summary['micro_benchmarks'][r.name] = r.to_dict()
                else:
                    results_summary['micro_benchmarks'][result.name] = result.to_dict()
            except Exception as e:
                print(f"Error in {name}: {e}")
                traceback.print_exc()
        
        # Benchmark device transfers separately
        try:
            transfer_results = self.benchmark_tensor_device_transfer()
            for result in transfer_results:
                results_summary['micro_benchmarks'][result.name] = result.to_dict()
        except Exception as e:
            print(f"Error in device transfer benchmark: {e}")
            
        # Run macro-benchmarks
        print("\n" + "="*40)
        print("MACRO-BENCHMARKS")
        print("="*40)
        
        for size in sizes:
            try:
                result = self.benchmark_training_iteration(size)
                results_summary['macro_benchmarks'][result.name] = result.to_dict()
            except Exception as e:
                print(f"Error in training iteration benchmark (size {size}): {e}")
                traceback.print_exc()
        
        # Run profiling
        if include_profiling:
            print("\n" + "="*40)
            print("PROFILING")
            print("="*40)
            
            try:
                bottlenecks = self.profile_training_iteration(
                    size='M',
                    profile_file=f'profile_detailed_{int(time.time())}.txt'
                )
                results_summary['bottlenecks'] = bottlenecks
            except Exception as e:
                print(f"Error in profiling: {e}")
                traceback.print_exc()
        
        # Save results
        timestamp = int(time.time())
        results_file = os.path.join(self.results_dir, f'benchmark_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        # Print summary
        print(f"Results saved to: {results_file}")
        print(f"Total benchmarks run: {len(results_summary['micro_benchmarks']) + len(results_summary['macro_benchmarks'])}")
        
        # Show key results
        key_benchmarks = ['state_encoding', 'cfr_single_traversal', 'full_training_iteration_size_M']
        print(f"\nKey Performance Metrics:")
        print("-" * 40)
        
        all_results = {**results_summary['micro_benchmarks'], **results_summary['macro_benchmarks']}
        for key in key_benchmarks:
            if key in all_results:
                result = all_results[key]
                print(f"{key:30s}: {result['mean_time']*1000:8.2f}ms ± {result['std_time']*1000:.2f}ms")
        
        return results_summary


def main():
    """Main benchmark runner with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepCFR Performance Benchmarker')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to run benchmarks on')
    parser.add_argument('--sizes', nargs='+', choices=['S', 'M', 'L'], default=['S', 'M'],
                       help='Benchmark sizes to run')
    parser.add_argument('--no-profile', action='store_true',
                       help='Skip profiling step')
    parser.add_argument('--warmup-runs', type=int, default=3,
                       help='Number of warmup runs')
    parser.add_argument('--benchmark-runs', type=int, default=10,
                       help='Number of benchmark runs to average')
    parser.add_argument('--results-dir', default='bench/results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    benchmarker = PerformanceBenchmarker(
        device=args.device,
        results_dir=args.results_dir,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs
    )
    
    results = benchmarker.run_all_benchmarks(
        include_profiling=not args.no_profile,
        sizes=args.sizes
    )
    
    return results


if __name__ == '__main__':
    main()