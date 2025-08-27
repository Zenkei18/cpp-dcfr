#!/usr/bin/env python3
"""
Benchmark comparison script to test original vs optimized DeepCFR performance.
This script compares performance with SPEED_REFAC=0 vs SPEED_REFAC=1.
"""

import os
import sys
import time
import json
import statistics
import subprocess
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import required modules
import pokers as pkrs
from src.core.model import encode_state as original_encode_state
from src.core.optimized_model import encode_state_optimized
from src.core.deep_cfr import DeepCFRAgent as OriginalDeepCFRAgent
from src.core.optimized_deep_cfr import OptimizedDeepCFRAgent
from src.agents.random_agent import RandomAgent


class PerformanceComparator:
    """Compare performance between original and optimized implementations."""
    
    def __init__(self, device='cpu', runs=5, warmup=2):
        self.device = device
        self.runs = runs
        self.warmup = warmup
        self.results = {'original': {}, 'optimized': {}}
        
        print(f"Performance Comparator initialized")
        print(f"Device: {device}")
        print(f"Warmup runs: {warmup}, Benchmark runs: {runs}")
        print("-" * 60)

    def timer(self, func, *args, **kwargs):
        """Time a function execution."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        end = time.perf_counter()
        return result, end - start

    def benchmark_function(self, name: str, original_func, optimized_func, *args, **kwargs):
        """Benchmark both original and optimized versions of a function."""
        print(f"\nBenchmarking {name}...")
        
        # Benchmark original version
        print("  Testing original implementation...")
        original_times = []
        
        # Warmup
        for _ in range(self.warmup):
            try:
                _, _ = self.timer(original_func, *args, **kwargs)
            except Exception as e:
                print(f"    Warning: Error in original warmup: {e}")
        
        # Benchmark runs
        for i in range(self.runs):
            try:
                result, duration = self.timer(original_func, *args, **kwargs)
                original_times.append(duration)
            except Exception as e:
                print(f"    Warning: Error in original run {i+1}: {e}")
        
        # Benchmark optimized version with SPEED_REFAC=1
        print("  Testing optimized implementation...")
        optimized_times = []
        
        # Set optimization flag
        old_speed_refac = os.environ.get('SPEED_REFAC', '0')
        os.environ['SPEED_REFAC'] = '1'
        
        try:
            # Need to reload modules to pick up the environment variable
            import importlib
            import src.core.optimized_model
            import src.core.optimized_deep_cfr
            importlib.reload(src.core.optimized_model)
            importlib.reload(src.core.optimized_deep_cfr)
            
            # Warmup
            for _ in range(self.warmup):
                try:
                    _, _ = self.timer(optimized_func, *args, **kwargs)
                except Exception as e:
                    print(f"    Warning: Error in optimized warmup: {e}")
            
            # Benchmark runs
            for i in range(self.runs):
                try:
                    result, duration = self.timer(optimized_func, *args, **kwargs)
                    optimized_times.append(duration)
                except Exception as e:
                    print(f"    Warning: Error in optimized run {i+1}: {e}")
        finally:
            # Restore original environment
            os.environ['SPEED_REFAC'] = old_speed_refac
        
        # Calculate statistics
        if original_times and optimized_times:
            original_mean = statistics.mean(original_times)
            original_std = statistics.stdev(original_times) if len(original_times) > 1 else 0
            
            optimized_mean = statistics.mean(optimized_times)
            optimized_std = statistics.stdev(optimized_times) if len(optimized_times) > 1 else 0
            
            speedup = original_mean / optimized_mean if optimized_mean > 0 else 0
            
            # Store results
            self.results['original'][name] = {
                'mean_time': original_mean,
                'std_time': original_std,
                'times': original_times
            }
            
            self.results['optimized'][name] = {
                'mean_time': optimized_mean,
                'std_time': optimized_std, 
                'times': optimized_times
            }
            
            print(f"  Original:  {original_mean*1000:.2f}ms ± {original_std*1000:.2f}ms")
            print(f"  Optimized: {optimized_mean*1000:.2f}ms ± {optimized_std*1000:.2f}ms")
            print(f"  Speedup:   {speedup:.2f}x {'✓' if speedup >= 1.5 else '⚠️' if speedup >= 1.1 else '❌'}")
            
            return speedup
        else:
            print(f"  Error: No successful runs for {name}")
            return 0

    def benchmark_state_encoding(self):
        """Benchmark state encoding performance."""
        # Create test state
        state = pkrs.State.from_seed(
            n_players=6,
            button=0,
            sb=1,
            bb=2,
            stake=200.0,
            seed=42
        )
        
        def original_encode(state, player_id):
            return original_encode_state(state, player_id)
        
        def optimized_encode(state, player_id):
            return encode_state_optimized(state, player_id)
        
        return self.benchmark_function(
            'state_encoding',
            original_encode,
            optimized_encode,
            state, 0
        )

    def benchmark_agent_creation(self):
        """Benchmark agent creation and initialization."""
        def create_original():
            return OriginalDeepCFRAgent(player_id=0, num_players=6, device=self.device)
        
        def create_optimized():
            return OptimizedDeepCFRAgent(player_id=0, num_players=6, device=self.device)
        
        return self.benchmark_function(
            'agent_creation',
            create_original,
            create_optimized
        )

    def benchmark_cfr_traversal(self):
        """Benchmark a single CFR traversal."""
        # Setup agents
        original_agent = OriginalDeepCFRAgent(player_id=0, num_players=6, device=self.device)
        
        # Create optimized agent with SPEED_REFAC=1
        old_speed_refac = os.environ.get('SPEED_REFAC', '0')
        os.environ['SPEED_REFAC'] = '1'
        
        try:
            import importlib
            import src.core.optimized_deep_cfr
            importlib.reload(src.core.optimized_deep_cfr)
            from src.core.optimized_deep_cfr import OptimizedDeepCFRAgent
            optimized_agent = OptimizedDeepCFRAgent(player_id=0, num_players=6, device=self.device)
        finally:
            os.environ['SPEED_REFAC'] = old_speed_refac
        
        # Create test state and random agents
        state = pkrs.State.from_seed(
            n_players=6,
            button=0,
            sb=1,
            bb=2,
            stake=200.0,
            seed=42
        )
        
        random_agents = [RandomAgent(i) for i in range(6)]
        
        def original_traverse():
            return original_agent.cfr_traverse(state, 1, random_agents)
        
        def optimized_traverse():
            os.environ['SPEED_REFAC'] = '1'
            try:
                return optimized_agent.cfr_traverse(state, 1, random_agents)
            finally:
                os.environ['SPEED_REFAC'] = old_speed_refac
        
        return self.benchmark_function(
            'cfr_traversal',
            original_traverse,
            optimized_traverse
        )

    def benchmark_network_training(self):
        """Benchmark network training performance."""
        # Create agents and populate some memory
        original_agent = OriginalDeepCFRAgent(player_id=0, num_players=6, device=self.device)
        
        old_speed_refac = os.environ.get('SPEED_REFAC', '0') 
        os.environ['SPEED_REFAC'] = '1'
        
        try:
            import importlib
            import src.core.optimized_deep_cfr
            importlib.reload(src.core.optimized_deep_cfr)
            from src.core.optimized_deep_cfr import OptimizedDeepCFRAgent
            optimized_agent = OptimizedDeepCFRAgent(player_id=0, num_players=6, device=self.device)
        finally:
            os.environ['SPEED_REFAC'] = old_speed_refac
        
        # Populate memory with sample data
        for i in range(200):
            experience = (
                np.random.randn(500),  # state
                np.random.randn(20),   # opponent features
                np.random.randint(0, 3),  # action
                np.random.rand(),      # bet size
                np.random.randn()      # regret
            )
            original_agent.advantage_memory.add(experience, priority=abs(np.random.randn()) + 0.1)
            optimized_agent.advantage_memory.add(experience, priority=abs(np.random.randn()) + 0.1)
        
        def original_train():
            return original_agent.train_advantage_network(batch_size=64, epochs=1)
        
        def optimized_train():
            os.environ['SPEED_REFAC'] = '1'
            try:
                return optimized_agent.train_advantage_network(batch_size=64, epochs=1)
            finally:
                os.environ['SPEED_REFAC'] = old_speed_refac
        
        return self.benchmark_function(
            'advantage_training',
            original_train,
            optimized_train
        )

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark comparing all components."""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE DEEPCFR PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        speedups = {}
        
        benchmarks = [
            ('state_encoding', self.benchmark_state_encoding),
            ('agent_creation', self.benchmark_agent_creation), 
            ('cfr_traversal', self.benchmark_cfr_traversal),
            ('advantage_training', self.benchmark_network_training),
        ]
        
        for name, benchmark_func in benchmarks:
            try:
                speedup = benchmark_func()
                speedups[name] = speedup
            except Exception as e:
                print(f"Error in {name} benchmark: {e}")
                import traceback
                traceback.print_exc()
                speedups[name] = 0
        
        # Save detailed results
        timestamp = int(time.time())
        results_file = f"bench/results/comparison_results_{timestamp}.json"
        
        full_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': self.device,
            'pytorch_version': torch.__version__,
            'speedups': speedups,
            'detailed_results': self.results
        }
        
        os.makedirs("bench/results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Results saved to: {results_file}")
        
        overall_speedup = statistics.geometric_mean([s for s in speedups.values() if s > 0]) if speedups else 0
        
        print(f"\nSpeedup Results:")
        print("-" * 40)
        for name, speedup in speedups.items():
            status = "✓ PASS" if speedup >= 1.5 else "⚠️  PARTIAL" if speedup >= 1.1 else "❌ FAIL"
            print(f"{name:20s}: {speedup:5.2f}x  {status}")
        
        print(f"\nOverall geometric mean speedup: {overall_speedup:.2f}x")
        
        # Check if we meet the 1.5x target
        target_met = overall_speedup >= 1.5
        print(f"Target (≥1.5x speedup): {'✓ ACHIEVED' if target_met else '❌ NOT MET'}")
        
        if target_met:
            print("\n🎉 Performance optimization successful!")
        else:
            print(f"\n⚠️  Performance target not fully met. Need {1.5/overall_speedup:.2f}x more improvement.")
        
        return full_results

def main():
    """Main benchmark comparison runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepCFR Performance Comparison')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to run benchmarks on')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of benchmark runs to average')
    parser.add_argument('--warmup', type=int, default=2,
                       help='Number of warmup runs')
    
    args = parser.parse_args()
    
    comparator = PerformanceComparator(
        device=args.device,
        runs=args.runs,
        warmup=args.warmup
    )
    
    results = comparator.run_comprehensive_benchmark()
    return results

if __name__ == '__main__':
    main()