#!/usr/bin/env python3
"""
Fixed training benchmark to test the actual training performance improvements.
This addresses the tensor shape issues found in the comparison benchmark.
"""

import os
import sys
import time
import statistics
import numpy as np
import torch

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pokers as pkrs
from src.core.deep_cfr import DeepCFRAgent as OriginalAgent
from src.core.optimized_deep_cfr import OptimizedDeepCFRAgent
from src.agents.random_agent import RandomAgent

class TrainingBenchmarker:
    """Specialized benchmarker for training performance."""
    
    def __init__(self, device='cpu', runs=3, warmup=1):
        self.device = device
        self.runs = runs
        self.warmup = warmup
    
    def benchmark_training_iteration(self):
        """Benchmark a complete training iteration with proper setup."""
        print("Setting up training iteration benchmark...")
        
        # Create agents with proper configuration
        original_agent = OriginalAgent(player_id=0, num_players=6, memory_size=1000, device=self.device)
        
        # Enable optimizations for the optimized agent
        old_env = os.environ.get('SPEED_REFAC', '0')
        os.environ['SPEED_REFAC'] = '1'
        
        # Reload modules to ensure optimization flag is picked up
        import importlib
        import src.core.optimized_model
        import src.core.optimized_deep_cfr
        importlib.reload(src.core.optimized_model)
        importlib.reload(src.core.optimized_deep_cfr)
        
        optimized_agent = OptimizedDeepCFRAgent(player_id=0, num_players=6, memory_size=1000, device=self.device)
        
        # Restore environment
        os.environ['SPEED_REFAC'] = old_env
        
        # Create random agents
        random_agents = [RandomAgent(i) for i in range(6)]
        
        def run_original_iteration():
            """Run a training iteration with the original agent."""
            iteration = 1
            
            # Run some traversals
            for t in range(10):  # Small number for benchmarking
                state = pkrs.State.from_seed(
                    n_players=6,
                    button=t % 6,
                    sb=1,
                    bb=2,
                    stake=200.0,
                    seed=t + 100
                )
                original_agent.cfr_traverse(state, iteration, random_agents)
            
            # Train networks if we have enough data
            if len(original_agent.advantage_memory) >= 32:
                adv_loss = original_agent.train_advantage_network(batch_size=32, epochs=1)
            else:
                adv_loss = 0
                
            if len(original_agent.strategy_memory) >= 32:
                strat_loss = original_agent.train_strategy_network(batch_size=32, epochs=1)
            else:
                strat_loss = 0
                
            return adv_loss, strat_loss
        
        def run_optimized_iteration():
            """Run a training iteration with the optimized agent."""
            # Temporarily enable optimizations
            os.environ['SPEED_REFAC'] = '1'
            
            try:
                iteration = 1
                
                # Run some traversals
                for t in range(10):  # Same number as original
                    state = pkrs.State.from_seed(
                        n_players=6,
                        button=t % 6,
                        sb=1,
                        bb=2,
                        stake=200.0,
                        seed=t + 200  # Different seeds to avoid cache effects
                    )
                    optimized_agent.cfr_traverse(state, iteration, random_agents)
                
                # Train networks if we have enough data
                if len(optimized_agent.advantage_memory) >= 32:
                    adv_loss = optimized_agent.train_advantage_network(batch_size=32, epochs=1)
                else:
                    adv_loss = 0
                    
                if len(optimized_agent.strategy_memory) >= 32:
                    strat_loss = optimized_agent.train_strategy_network(batch_size=32, epochs=1)
                else:
                    strat_loss = 0
                    
                return adv_loss, strat_loss
            finally:
                os.environ['SPEED_REFAC'] = old_env
        
        # Benchmark both versions
        print("Benchmarking original training iteration...")
        original_times = []
        
        # Warmup
        for _ in range(self.warmup):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                run_original_iteration()
            except Exception as e:
                print(f"Warning: Original warmup error: {e}")
            if self.device == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark runs
        for i in range(self.runs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                run_original_iteration()
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                original_times.append(end - start)
                print(f"  Original run {i+1}: {(end-start)*1000:.1f}ms")
            except Exception as e:
                print(f"Warning: Original run {i+1} error: {e}")
        
        print("Benchmarking optimized training iteration...")
        optimized_times = []
        
        # Warmup  
        for _ in range(self.warmup):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                run_optimized_iteration()
            except Exception as e:
                print(f"Warning: Optimized warmup error: {e}")
            if self.device == 'cuda':
                torch.cuda.synchronize()
        
        # Benchmark runs
        for i in range(self.runs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                run_optimized_iteration()
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                optimized_times.append(end - start)
                print(f"  Optimized run {i+1}: {(end-start)*1000:.1f}ms")
            except Exception as e:
                print(f"Warning: Optimized run {i+1} error: {e}")
        
        # Calculate results
        if original_times and optimized_times:
            original_mean = statistics.mean(original_times)
            optimized_mean = statistics.mean(optimized_times)
            speedup = original_mean / optimized_mean if optimized_mean > 0 else 0
            
            print(f"\nTraining Iteration Results:")
            print(f"Original:  {original_mean*1000:.1f}ms")
            print(f"Optimized: {optimized_mean*1000:.1f}ms")
            print(f"Speedup:   {speedup:.2f}x {'✓' if speedup >= 1.5 else '⚠️' if speedup >= 1.1 else '❌'}")
            
            return speedup
        else:
            print("No successful runs to compare")
            return 0

def main():
    benchmarker = TrainingBenchmarker(device='cpu', runs=3, warmup=1)
    speedup = benchmarker.benchmark_training_iteration()
    
    if speedup >= 1.5:
        print(f"\n✅ Training performance target achieved: {speedup:.2f}x speedup")
    else:
        print(f"\n⚠️ Training performance target not fully met: {speedup:.2f}x speedup (target: 1.5x)")
    
    return speedup

if __name__ == '__main__':
    main()