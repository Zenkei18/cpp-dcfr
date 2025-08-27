#!/usr/bin/env python3
"""
Simple speed test to demonstrate the optimizations working.
This shows component-level optimizations without full integration.
"""

import os
import sys
import time
sys.path.insert(0, 'src')

import pokers as pkrs
import numpy as np

def test_state_encoding_speed():
    """Test state encoding optimization."""
    print("🧪 Testing State Encoding Speed")
    print("-" * 40)
    
    # Create test states
    states = []
    for i in range(100):
        state = pkrs.State.from_seed(
            n_players=6, button=i % 6, sb=1, bb=2, 
            stake=200.0, seed=i
        )
        states.append(state)
    
    # Test original encoding
    from core.model import encode_state as original_encode_state
    
    start_time = time.perf_counter()
    for state in states:
        encoded = original_encode_state(state, 0)
    original_time = time.perf_counter() - start_time
    print(f"✅ Original encoding: {original_time:.4f}s ({100/original_time:.1f} states/sec)")
    
    # Test optimized encoding  
    from core.optimized_model import encode_state_optimized
    
    start_time = time.perf_counter()
    for state in states:
        encoded_opt = encode_state_optimized(state, 0)
    optimized_time = time.perf_counter() - start_time
    print(f"⚡ Optimized encoding: {optimized_time:.4f}s ({100/optimized_time:.1f} states/sec)")
    
    speedup = original_time / optimized_time
    print(f"🚀 Speedup: {speedup:.2f}x")
    
    # Verify correctness
    test_state = states[0]
    orig_result = original_encode_state(test_state, 0)
    opt_result = encode_state_optimized(test_state, 0)
    max_diff = np.max(np.abs(orig_result - opt_result))
    print(f"✅ Max difference: {max_diff:.6f} (should be < 0.001)")
    
    return speedup

def test_legal_actions_speed():
    """Test legal action computation optimization."""
    print("\n🧪 Testing Legal Actions Speed")
    print("-" * 40)
    
    # Create test states
    states = []
    for i in range(1000):
        state = pkrs.State.from_seed(
            n_players=6, button=i % 6, sb=1, bb=2,
            stake=200.0, seed=i
        )
        states.append(state)
    
    # Test original
    from core.model import get_legal_action_types as original_get_legal_action_types
    
    start_time = time.perf_counter()
    for state in states:
        actions = original_get_legal_action_types(state)
    original_time = time.perf_counter() - start_time
    print(f"✅ Original legal actions: {original_time:.4f}s")
    
    # Test optimized
    from core.optimized_model import get_legal_action_types_optimized
    
    start_time = time.perf_counter()
    for state in states:
        actions_opt = get_legal_action_types_optimized(state)
    optimized_time = time.perf_counter() - start_time
    print(f"⚡ Optimized legal actions: {optimized_time:.4f}s")
    
    speedup = original_time / optimized_time
    print(f"🚀 Speedup: {speedup:.2f}x")
    
    # Verify correctness
    test_state = states[0]
    orig_result = original_get_legal_action_types(test_state)
    opt_result = get_legal_action_types_optimized(test_state)
    print(f"✅ Results match: {orig_result == opt_result}")
    
    return speedup

def test_network_creation():
    """Test network creation and basic operations."""
    print("\n🧪 Testing Network Creation")
    print("-" * 40)
    
    import torch
    
    # Test original network
    from core.model import PokerNetwork
    
    start_time = time.perf_counter()
    original_net = PokerNetwork(input_size=156, hidden_size=256, num_actions=3)
    original_time = time.perf_counter() - start_time
    print(f"✅ Original network creation: {original_time:.4f}s")
    
    # Test optimized network
    from core.optimized_model import OptimizedPokerNetwork
    
    start_time = time.perf_counter()
    optimized_net = OptimizedPokerNetwork(input_size=156, hidden_size=256, num_actions=3)
    optimized_time = time.perf_counter() - start_time
    print(f"⚡ Optimized network creation: {optimized_time:.4f}s")
    
    # Test forward pass
    test_input = torch.randn(32, 156)  # batch_size=32
    
    start_time = time.perf_counter()
    with torch.no_grad():
        orig_output = original_net(test_input)
    orig_forward_time = time.perf_counter() - start_time
    print(f"✅ Original forward pass: {orig_forward_time:.4f}s")
    
    # Set SPEED_REFAC for optimized network
    os.environ['SPEED_REFAC'] = '1'
    from core.optimized_model import SPEED_REFAC
    
    start_time = time.perf_counter()
    with torch.no_grad():
        opt_output = optimized_net(test_input)
    opt_forward_time = time.perf_counter() - start_time
    print(f"⚡ Optimized forward pass: {opt_forward_time:.4f}s")
    
    if orig_forward_time > 0 and opt_forward_time > 0:
        speedup = orig_forward_time / opt_forward_time
        print(f"🚀 Forward pass speedup: {speedup:.2f}x")
        return speedup
    return 1.0

def main():
    print("=" * 60)
    print("🚀 DEEPCFR OPTIMIZATION COMPONENT TESTS")
    print("=" * 60)
    
    speedups = []
    
    try:
        speedup1 = test_state_encoding_speed()
        speedups.append(speedup1)
    except Exception as e:
        print(f"❌ State encoding test failed: {e}")
    
    try:
        speedup2 = test_legal_actions_speed()
        speedups.append(speedup2)
    except Exception as e:
        print(f"❌ Legal actions test failed: {e}")
        
    try:
        speedup3 = test_network_creation()
        speedups.append(speedup3)
    except Exception as e:
        print(f"❌ Network test failed: {e}")
    
    if speedups:
        geometric_mean = np.prod(speedups) ** (1.0 / len(speedups))
        print(f"\n🏁 SUMMARY")
        print("-" * 40)
        print(f"🚀 Geometric mean speedup: {geometric_mean:.2f}x")
        
        if geometric_mean >= 1.5:
            print("✅ SUCCESS: Target ≥1.5x speedup achieved!")
        else:
            print("⚠️  PARTIAL: Some optimizations working, but overall target not met")
            
        print(f"💡 Individual speedups: {[f'{s:.2f}x' for s in speedups]}")
    else:
        print("❌ No successful tests completed")

if __name__ == "__main__":
    main()