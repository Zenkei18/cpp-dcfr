#!/usr/bin/env python3
"""
Optimized training script that can switch between original and optimized implementations.
Set SPEED_REFAC=1 to enable optimizations.
"""

import os
import sys
import pokers as pkrs
import numpy as np
import random
import torch
import time
import matplotlib.pyplot as plt
import argparse

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Check if optimizations are enabled
SPEED_REFAC = os.environ.get('SPEED_REFAC', '0') == '1'

print(f"🚀 Training with optimizations: {'ENABLED' if SPEED_REFAC else 'DISABLED'}")
print(f"   Set SPEED_REFAC=1 to enable optimizations")

# Import appropriate implementations based on flag
if SPEED_REFAC:
    from src.core.optimized_deep_cfr import OptimizedDeepCFRAgent as DeepCFRAgent
    from src.core.model import set_verbose, encode_state  # Use original model functions
    print("   Using optimized implementations ⚡")
else:
    from src.core.deep_cfr import DeepCFRAgent
    from src.core.model import set_verbose, encode_state
    print("   Using original implementations")

from src.utils.logging import log_game_error
from src.utils.settings import STRICT_CHECKING, set_strict_checking
from src.agents.random_agent import RandomAgent

def evaluate_against_random(agent, num_games=500, num_players=6):
    """Evaluate the trained agent against random opponents."""
    random_agents = [RandomAgent(i) for i in range(num_players)]
    total_profit = 0
    completed_games = 0
    
    for game in range(num_games):
        try:
            # Create a new poker game
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=game % num_players,  # Rotate button for fairness
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 2147483647)  # Random evaluation games
            )
            
            # Play until the game is over
            while not state.final_state:
                current_player = state.current_player
                
                if current_player == agent.player_id:
                    action = agent.choose_action(state)
                else:
                    action = random_agents[current_player].choose_action(state)
                
                # Apply the action with conditional status check
                new_state = state.apply_action(action)
                if new_state.status != pkrs.StateStatus.Ok:
                    log_file = log_game_error(state, action, f"State status not OK ({new_state.status})")
                    if STRICT_CHECKING:
                        raise ValueError(f"State status not OK ({new_state.status}). Details logged to {log_file}")
                    else:
                        print(f"WARNING: State status not OK ({new_state.status}) in game {game}. Details logged to {log_file}")
                        break  # Skip this game in non-strict mode
                
                state = new_state
            
            # Only count completed games
            if state.final_state:
                # Add the profit for this game
                profit = state.players_state[agent.player_id].reward
                total_profit += profit
                completed_games += 1
                
        except Exception as e:
            if STRICT_CHECKING:
                raise  # Re-raise the exception in strict mode
            else:
                print(f"Error in game {game}: {e}")
                # Continue with next game in non-strict mode
    
    # Return average profit only for completed games
    if completed_games == 0:
        print("WARNING: No games completed during evaluation!")
        return 0
    
    return total_profit / completed_games

def train_deep_cfr(num_iterations=100, traversals_per_iteration=50, 
                   num_players=6, player_id=0, save_dir="models", 
                   log_dir="logs/deepcfr", verbose=False):
    """
    Train a Deep CFR agent in a 6-player No-Limit Texas Hold'em game
    against 5 random opponents.
    """
    # Import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create the directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Initialize the agent
    print(f"🤖 Creating {'Optimized' if SPEED_REFAC else 'Original'} DeepCFR Agent...")
    agent = DeepCFRAgent(player_id=player_id, num_players=num_players, 
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create random agents for the opponents
    random_agents = [RandomAgent(i) for i in range(num_players)]
    
    # For tracking learning progress
    losses = []
    profits = []
    
    # Initial evaluation before training begins
    print("📊 Initial evaluation...")
    start_time = time.time()
    initial_profit = evaluate_against_random(agent, num_games=100, num_players=num_players)
    eval_time = time.time() - start_time
    profits.append(initial_profit)
    print(f"Initial average profit per game: {initial_profit:.2f} (took {eval_time:.1f}s)")
    writer.add_scalar('Performance/Profit', initial_profit, 0)
    
    # Checkpoint frequency
    checkpoint_frequency = 10  # Save more frequently for testing
    
    # Training loop
    print(f"🎯 Starting training for {num_iterations} iterations...")
    for iteration in range(1, num_iterations + 1):
        agent.iteration_count = iteration
        start_time = time.time()
        
        print(f"Iteration {iteration}/{num_iterations}")
        
        # Run traversals to collect data
        print("  📈 Collecting data...")
        traversal_start = time.time()
        for t in range(traversals_per_iteration):
            # Create a new poker game
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=random.randint(0, num_players-1),
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 2147483647)  # Full 32-bit range for maximum variety
            )
            
            # Perform CFR traversal
            agent.cfr_traverse(state, iteration, random_agents)
        
        traversal_time = time.time() - traversal_start
        writer.add_scalar('Time/Traversal', traversal_time, iteration)
        print(f"  ⚡ CFR traversals: {traversal_time:.2f}s ({traversals_per_iteration} traversals)")
        
        # Train advantage network
        print("  🧠 Training advantage network...")
        train_start = time.time()
        adv_loss = agent.train_advantage_network(batch_size=64, epochs=1)
        train_time = time.time() - train_start
        losses.append(adv_loss)
        print(f"  Advantage network loss: {adv_loss:.6f} (took {train_time:.2f}s)")
        
        # Log the loss to tensorboard
        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        writer.add_scalar('Memory/Advantage', len(agent.advantage_memory), iteration)
        writer.add_scalar('Time/Training', train_time, iteration)
        
        # Every few iterations, train the strategy network and evaluate
        if iteration % 5 == 0 or iteration == num_iterations:
            print("  🎯 Training strategy network...")
            strat_start = time.time()
            strat_loss = agent.train_strategy_network(batch_size=64, epochs=1)
            strat_time = time.time() - strat_start
            print(f"  Strategy network loss: {strat_loss:.6f} (took {strat_time:.2f}s)")
            writer.add_scalar('Loss/Strategy', strat_loss, iteration)
            
            # Evaluate the agent
            print("  📊 Evaluating agent...")
            eval_start = time.time()
            avg_profit = evaluate_against_random(agent, num_games=100, num_players=num_players)
            eval_time = time.time() - eval_start
            profits.append(avg_profit)
            print(f"  Average profit per game: {avg_profit:.2f} (took {eval_time:.1f}s)")
            writer.add_scalar('Performance/Profit', avg_profit, iteration)
        
        # Show performance stats if optimizations are enabled
        if SPEED_REFAC and hasattr(agent, 'get_performance_stats'):
            if iteration % 10 == 0:
                stats = agent.get_performance_stats()
                if stats and 'encode_state' in stats:
                    encode_stats = stats['encode_state']
                    print(f"  📈 State encoding: {encode_stats['time_per_call']:.3f}ms avg per call")
        
        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        print(f"  ✅ Iteration completed in {elapsed:.2f}s")
        print(f"  Memory: Advantage={len(agent.advantage_memory)}, Strategy={len(agent.strategy_memory)}")
        writer.add_scalar('Memory/Strategy', len(agent.strategy_memory), iteration)
        
        # Commit the tensorboard logs
        writer.flush()
        print()
    
    # Final evaluation
    print("🏁 Final evaluation...")
    final_start = time.time()
    avg_profit = evaluate_against_random(agent, num_games=200)
    final_time = time.time() - final_start
    print(f"Final performance: Average profit per game: {avg_profit:.2f} (took {final_time:.1f}s)")
    writer.add_scalar('Performance/FinalProfit', avg_profit, 0)
    
    # Show final performance stats
    if SPEED_REFAC and hasattr(agent, 'get_performance_stats'):
        print(f"\n📊 Performance Statistics:")
        stats = agent.get_performance_stats()
        for func_name, func_stats in stats.items():
            print(f"  {func_name}: {func_stats['time_per_call']:.3f}ms avg ({func_stats['call_count']} calls)")
    
    # Close the tensorboard writer
    writer.close()
    
    return agent, losses, profits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent for poker (with optimizations)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=100, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=50, help='Traversals per iteration')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/deepcfr_optimized', help='Directory for tensorboard logs')
    parser.add_argument('--strict', action='store_true', help='Enable strict error checking')
    args = parser.parse_args()

    # Show optimization status
    print(f"\n{'='*60}")
    print("🚀 DEEPCFR OPTIMIZED TRAINING")
    print(f"{'='*60}")
    print(f"Optimizations: {'✅ ENABLED' if SPEED_REFAC else '❌ DISABLED'}")
    if not SPEED_REFAC:
        print("💡 To enable optimizations: export SPEED_REFAC=1")
    print(f"{'='*60}\n")

    # Strict training for debug
    set_strict_checking(args.strict)
    
    print(f"Starting Deep CFR training for {args.iterations} iterations")
    print(f"Using {args.traversals} traversals per iteration")
    print(f"Logs will be saved to: {args.log_dir}")
    print(f"Models will be saved to: {args.save_dir}")
    
    # Train the Deep CFR agent
    agent, losses, profits = train_deep_cfr(
        num_iterations=args.iterations,
        traversals_per_iteration=args.traversals,
        num_players=6,
        player_id=0,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        verbose=args.verbose
    )
    
    print("\n🎉 Training Summary:")
    print(f"Final loss: {losses[-1]:.6f}")
    if profits:
        print(f"Final average profit vs random: {profits[-1]:.2f}")
    
    print(f"\nTo view training progress:")
    print(f"Run: tensorboard --logdir={args.log_dir}")
    print("Then open http://localhost:6006 in your browser")