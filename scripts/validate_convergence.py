# scripts/validate_convergence.py
"""
Validation script to test convergence improvements.
This script runs short training sessions to validate that enhanced configurations
achieve the required performance gates.
"""

import sys
import os
import time
import torch
import numpy as np
import random
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.enhanced_deep_cfr import EnhancedDeepCFRAgent
from src.core.deep_cfr import DeepCFRAgent
from src.utils.config import load_config
from src.agents.random_agent import RandomAgent
from src.core.model import set_verbose
import pokers as pkrs


def quick_evaluate(agent, num_games=100):
    """Quick evaluation against random agents."""
    random_agents = [RandomAgent(i) for i in range(6)]
    total_profit = 0
    completed_games = 0
    
    for game in range(num_games):
        try:
            state = pkrs.State.from_seed(
                n_players=6,
                button=game % 6,
                sb=1,
                bb=2,
                stake=200.0,
                seed=game + 50000  # Different seed range
            )
            
            while not state.final_state:
                current_player = state.current_player
                
                if current_player == agent.player_id:
                    action = agent.choose_action(state)
                else:
                    action = random_agents[current_player].choose_action(state)
                
                new_state = state.apply_action(action)
                if new_state.status != pkrs.StateStatus.Ok:
                    break
                state = new_state
            
            if state.final_state:
                profit = state.players_state[agent.player_id].reward
                total_profit += profit
                completed_games += 1
                
        except Exception as e:
            continue
    
    if completed_games == 0:
        return 0
    return total_profit / completed_games


def train_short_session(agent, num_iterations=50, traversals_per_iteration=50):
    """Run a short training session."""
    random_agents = [RandomAgent(i) for i in range(6)]
    losses = []
    
    # Initial evaluation
    initial_profit = quick_evaluate(agent)
    
    start_time = time.time()
    
    for iteration in range(1, num_iterations + 1):
        agent.iteration_count = iteration
        
        # Collect data
        for _ in range(traversals_per_iteration):
            state = pkrs.State.from_seed(
                n_players=6,
                button=random.randint(0, 5),
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )
            
            if hasattr(agent, 'cfr_traverse'):
                agent.cfr_traverse(state, iteration, random_agents)
        
        # Train networks
        adv_loss = agent.train_advantage_network()
        losses.append(adv_loss)
        
        if iteration % 10 == 0:
            agent.train_strategy_network()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Final evaluation
    final_profit = quick_evaluate(agent)
    
    return {
        'initial_profit': initial_profit,
        'final_profit': final_profit,
        'improvement': (final_profit - initial_profit) / abs(initial_profit) * 100 if initial_profit != 0 else 0,
        'losses': losses,
        'training_time': training_time,
        'time_per_iteration': training_time / num_iterations
    }


def validate_baseline_vs_enhanced():
    """Compare baseline configuration with enhanced configurations."""
    set_verbose(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Neural Network Convergence Accelerator - Validation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Test configurations
    configs_to_test = [
        ('baseline', 'configs/ablate/baseline.yaml'),
        ('adamw_clip', 'configs/ablate/adamw_clip.yaml'),
        ('cosine_amp', 'configs/ablate/cosine_amp.yaml'),
        ('norm_targets', 'configs/ablate/norm_targets.yaml')
    ]
    
    results = {}
    
    for config_name, config_path in configs_to_test:
        print(f"Testing {config_name}...")
        
        try:
            # Load configuration
            config = load_config(config_path)
            
            # Create agent
            if config_name == 'baseline':
                # Use original agent for baseline
                agent = DeepCFRAgent(
                    player_id=0,
                    num_players=6,
                    device=device
                )
            else:
                # Use enhanced agent
                agent = EnhancedDeepCFRAgent(
                    config=config,
                    player_id=0,
                    num_players=6,
                    device=device
                )
            
            # Run short training session
            result = train_short_session(agent, num_iterations=50, traversals_per_iteration=25)
            results[config_name] = result
            
            print(f"  Initial profit: {result['initial_profit']:.2f}")
            print(f"  Final profit: {result['final_profit']:.2f}")
            print(f"  Improvement: {result['improvement']:.1f}%")
            print(f"  Training time: {result['training_time']:.1f}s")
            print(f"  Time per iteration: {result['time_per_iteration']:.2f}s")
            if result['losses']:
                print(f"  Final loss: {result['losses'][-1]:.6f}")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[config_name] = {'error': str(e)}
            print()
    
    # Analysis
    print("Performance Analysis")
    print("=" * 30)
    
    if 'baseline' in results and not 'error' in results['baseline']:
        baseline_result = results['baseline']
        baseline_improvement = baseline_result['improvement']
        baseline_time = baseline_result['time_per_iteration']
        
        print(f"Baseline improvement: {baseline_improvement:.1f}%")
        print(f"Baseline time/iter: {baseline_time:.2f}s")
        print()
        
        for config_name, result in results.items():
            if config_name == 'baseline' or 'error' in result:
                continue
                
            improvement_delta = result['improvement'] - baseline_improvement
            time_ratio = result['time_per_iteration'] / baseline_time
            time_regression = (time_ratio - 1) * 100
            
            print(f"{config_name}:")
            print(f"  Improvement delta: {improvement_delta:+.1f}%")
            print(f"  Time regression: {time_regression:+.1f}%")
            
            # Check gates
            convergence_gate = improvement_delta >= 20.0  # At least 20% better improvement
            time_gate = time_regression <= 10.0  # No more than 10% time regression
            
            print(f"  Convergence gate (≥+20%): {'PASS' if convergence_gate else 'FAIL'}")
            print(f"  Time gate (≤+10%): {'PASS' if time_gate else 'FAIL'}")
            print(f"  Overall: {'PASS' if convergence_gate and time_gate else 'FAIL'}")
            print()
        
    else:
        print("Baseline test failed - cannot perform comparison")
    
    return results


def validate_specific_improvements():
    """Test specific improvements in isolation."""
    print("Specific Improvement Validation")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test AdamW vs Adam
    print("1. Testing AdamW vs Adam...")
    
    # Create simple test agents
    from src.utils.config import Config, ModelConfig, TrainingConfig, MemoryConfig, LoggingConfig
    
    # Adam config
    adam_config = Config(
        model=ModelConfig(),
        training=TrainingConfig(optimizer="adam", advantage_lr=1e-4, strategy_lr=1e-4),
        memory=MemoryConfig(),
        logging=LoggingConfig()
    )
    
    # AdamW config  
    adamw_config = Config(
        model=ModelConfig(),
        training=TrainingConfig(optimizer="adamw", advantage_lr=1e-4, strategy_lr=1e-4, weight_decay=1e-2),
        memory=MemoryConfig(),
        logging=LoggingConfig()
    )
    
    try:
        adam_agent = EnhancedDeepCFRAgent(adam_config, device=device)
        adamw_agent = EnhancedDeepCFRAgent(adamw_config, device=device)
        
        adam_result = train_short_session(adam_agent, 30, 20)
        adamw_result = train_short_session(adamw_agent, 30, 20)
        
        print(f"  Adam improvement: {adam_result['improvement']:.1f}%")
        print(f"  AdamW improvement: {adamw_result['improvement']:.1f}%")
        print(f"  AdamW advantage: {adamw_result['improvement'] - adam_result['improvement']:+.1f}%")
        
    except Exception as e:
        print(f"  Error testing optimizers: {e}")
    
    print()
    
    # Test Layer Normalization
    print("2. Testing LayerNorm impact...")
    
    # Without LayerNorm
    no_norm_config = Config(
        model=ModelConfig(use_layer_norm=False),
        training=TrainingConfig(optimizer="adamw", advantage_lr=1e-4, strategy_lr=1e-4),
        memory=MemoryConfig(),
        logging=LoggingConfig()
    )
    
    # With LayerNorm
    with_norm_config = Config(
        model=ModelConfig(use_layer_norm=True),
        training=TrainingConfig(optimizer="adamw", advantage_lr=1e-4, strategy_lr=1e-4),
        memory=MemoryConfig(),
        logging=LoggingConfig()
    )
    
    try:
        no_norm_agent = EnhancedDeepCFRAgent(no_norm_config, device=device)
        with_norm_agent = EnhancedDeepCFRAgent(with_norm_config, device=device)
        
        no_norm_result = train_short_session(no_norm_agent, 30, 20)
        with_norm_result = train_short_session(with_norm_agent, 30, 20)
        
        print(f"  Without LayerNorm: {no_norm_result['improvement']:.1f}%")
        print(f"  With LayerNorm: {with_norm_result['improvement']:.1f}%")
        print(f"  LayerNorm advantage: {with_norm_result['improvement'] - no_norm_result['improvement']:+.1f}%")
        
    except Exception as e:
        print(f"  Error testing LayerNorm: {e}")
    
    print()


if __name__ == "__main__":
    # Ensure configs exist
    config_dir = Path("configs/ablate")
    if not config_dir.exists():
        print("Error: Configuration files not found. Please run the main training script first to generate configs.")
        sys.exit(1)
    
    print("Starting convergence validation...")
    print()
    
    # Run main validation
    results = validate_baseline_vs_enhanced()
    
    # Run specific tests
    validate_specific_improvements()
    
    print("Validation completed!")
    print()
    print("If tests pass, the neural network convergence accelerator is working correctly.")
    print("Run the full ablation study with: python src/training/train_enhanced.py --create-ablation")