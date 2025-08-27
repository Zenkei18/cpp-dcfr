#!/usr/bin/env python3
"""
Debug script to verify tensor dimensions and identify issues.
Run this on your local machine to debug the dimension mismatch.
"""

import os
import sys
sys.path.insert(0, 'src')
import torch
import numpy as np
import pokers as pkrs

def check_model_dimensions():
    """Check if model input dimensions are correct."""
    print("🔍 Checking Model Dimensions")
    print("-" * 40)
    
    # Test state encoding
    try:
        from core.model import encode_state
        state = pkrs.State.from_seed(n_players=6, button=0, sb=1, bb=2, stake=200.0, seed=42)
        encoded = encode_state(state, 0)
        print(f"✅ State encoding size: {len(encoded)}")
        actual_size = len(encoded)
    except Exception as e:
        print(f"❌ State encoding failed: {e}")
        return False
    
    # Test PokerNetwork
    try:
        from core.model import PokerNetwork
        
        # Check if using old default (500) vs new correct size (156)
        import inspect
        sig = inspect.signature(PokerNetwork.__init__)
        default_input_size = sig.parameters['input_size'].default
        print(f"PokerNetwork default input_size: {default_input_size}")
        
        if default_input_size != actual_size:
            print(f"❌ MISMATCH: Network expects {default_input_size}, but state encoding is {actual_size}")
            print("💡 Fix: Change PokerNetwork.__init__ input_size default to 156")
            return False
        else:
            print(f"✅ Network input size matches state encoding: {actual_size}")
            
    except Exception as e:
        print(f"❌ Network creation failed: {e}")
        return False
    
    # Test network forward pass
    try:
        net = PokerNetwork()
        test_input = torch.randn(32, actual_size)
        action_logits, bet_size = net(test_input)
        print(f"✅ Network forward pass: input{test_input.shape} -> actions{action_logits.shape}, bets{bet_size.shape}")
    except Exception as e:
        print(f"❌ Network forward pass failed: {e}")
        return False
        
    return True

def check_optimized_dimensions():
    """Check optimized model dimensions."""
    print("\n🔍 Checking Optimized Model Dimensions")
    print("-" * 40)
    
    try:
        from core.optimized_model import OptimizedPokerNetwork
        import inspect
        
        sig = inspect.signature(OptimizedPokerNetwork.__init__)
        default_input_size = sig.parameters['input_size'].default
        print(f"OptimizedPokerNetwork default input_size: {default_input_size}")
        
        # Test forward pass
        net = OptimizedPokerNetwork()
        test_input = torch.randn(32, default_input_size)
        action_logits, bet_size = net(test_input)
        print(f"✅ Optimized network: input{test_input.shape} -> actions{action_logits.shape}, bets{bet_size.shape}")
        
    except Exception as e:
        print(f"❌ Optimized network failed: {e}")
        return False
        
    return True

def test_gather_operation():
    """Test the specific gather operation that's failing."""
    print("\n🔍 Testing Gather Operation")
    print("-" * 40)
    
    # Simulate the failing operation
    batch_size = 64
    num_actions = 3
    
    action_advantages = torch.randn(batch_size, num_actions)
    action_type_tensors = torch.randint(0, num_actions, (batch_size,))
    
    print(f"action_advantages shape: {action_advantages.shape} (dims: {action_advantages.dim()})")
    print(f"action_type_tensors shape: {action_type_tensors.shape} (dims: {action_type_tensors.dim()})")
    print(f"action_type_tensors sample: {action_type_tensors[:10]}")
    
    try:
        # This is the line that's failing
        predicted_regrets = action_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
        print(f"✅ Gather operation successful: {predicted_regrets.shape}")
        return True
    except Exception as e:
        print(f"❌ Gather operation failed: {e}")
        print(f"action_advantages.dim(): {action_advantages.dim()}")
        print(f"action_type_tensors.unsqueeze(1).dim(): {action_type_tensors.unsqueeze(1).dim()}")
        return False

def check_batch_processor():
    """Check the batch processor tensor creation."""
    print("\n🔍 Testing Batch Processor")
    print("-" * 40)
    
    os.environ['SPEED_REFAC'] = '1'
    
    try:
        from core.optimized_model import OptimizedBatchProcessor
        
        processor = OptimizedBatchProcessor(device='cpu')
        
        # Create fake batch data
        fake_states = [np.random.randn(156) for _ in range(64)]
        fake_actions = [np.random.randint(0, 3) for _ in range(64)]
        fake_regrets = [np.random.randn() for _ in range(64)]
        fake_strategies = [np.random.randn(3) for _ in range(64)]
        fake_bet_sizes = [np.random.randn() for _ in range(64)]
        
        batch_data = list(zip(fake_states, fake_actions, fake_regrets, fake_strategies, fake_bet_sizes))
        
        tensors = processor.prepare_batch_tensors(batch_data, 64)
        
        print(f"✅ Batch processor created tensors:")
        for key, tensor in tensors.items():
            print(f"  {key}: {tensor.shape}")
            
    except Exception as e:
        print(f"❌ Batch processor failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

def main():
    print("=" * 60)
    print("🐛 DEEPCFR TENSOR DIMENSION DEBUGGER")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= check_model_dimensions()
    all_passed &= check_optimized_dimensions() 
    all_passed &= test_gather_operation()
    all_passed &= check_batch_processor()
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL CHECKS PASSED - Tensor dimensions are correct!")
        print("💡 If you're still getting errors, make sure to:")
        print("   1. git pull origin perf/cfr-speedups")
        print("   2. Check that PokerNetwork input_size=156 (not 500)")
    else:
        print("❌ SOME CHECKS FAILED - See errors above")
        print("💡 Most likely fix: Update PokerNetwork input_size from 500 to 156")
        
    print(f"{'='*60}")

if __name__ == "__main__":
    main()