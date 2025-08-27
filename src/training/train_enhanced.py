# src/training/train_enhanced.py
import os
import time
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pokers as pkrs

from src.core.enhanced_deep_cfr import EnhancedDeepCFRAgent
from src.core.model import set_verbose
from src.utils.config import load_config, save_config
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
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=game % num_players,
                sb=1,
                bb=2,
                stake=200.0,
                seed=game
            )
            
            while not state.final_state:
                current_player = state.current_player
                
                if current_player == agent.player_id:
                    action = agent.choose_action(state)
                else:
                    action = random_agents[current_player].choose_action(state)
                
                new_state = state.apply_action(action)
                if new_state.status != pkrs.StateStatus.Ok:
                    log_file = log_game_error(state, action, f"State status not OK ({new_state.status})")
                    if STRICT_CHECKING:
                        raise ValueError(f"State status not OK ({new_state.status}). Details logged to {log_file}")
                    else:
                        print(f"WARNING: State status not OK ({new_state.status}) in game {game}. Details logged to {log_file}")
                        break
                
                state = new_state
            
            if state.final_state:
                profit = state.players_state[agent.player_id].reward
                total_profit += profit
                completed_games += 1
                
        except Exception as e:
            if STRICT_CHECKING:
                raise
            else:
                print(f"Error in game {game}: {e}")
    
    if completed_games == 0:
        print("WARNING: No games completed during evaluation!")
        return 0
    
    return total_profit / completed_games


def create_training_report(agent, config, save_dir, final_profit):
    """Create a comprehensive training report with plots and statistics."""
    
    # Create docs directory if it doesn't exist
    docs_dir = os.path.join(save_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    stats = agent.get_training_stats()
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Training Report - {config.model.name}', fontsize=16)
    
    # Loss curves
    if stats['advantage_loss']:
        axes[0, 0].plot(stats['advantage_loss'], label='Advantage', alpha=0.8)
    if stats['strategy_loss']:
        axes[0, 0].plot(range(0, len(stats['strategy_loss']) * 10, 10), 
                       stats['strategy_loss'], label='Strategy', alpha=0.8)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate curves
    if stats['learning_rates']['advantage']:
        axes[0, 1].plot(stats['learning_rates']['advantage'], label='Advantage', alpha=0.8)
    if stats['learning_rates']['strategy']:
        axes[0, 1].plot(stats['learning_rates']['strategy'], label='Strategy', alpha=0.8)
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient norms
    if stats['grad_norms']['advantage']:
        axes[0, 2].plot(stats['grad_norms']['advantage'], label='Advantage', alpha=0.8)
    if stats['grad_norms']['strategy']:
        axes[0, 2].plot(stats['grad_norms']['strategy'], label='Strategy', alpha=0.8)
    axes[0, 2].set_title('Gradient Norms')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Gradient Norm')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Steps per second
    if stats['steps_per_sec']:
        axes[1, 0].plot(stats['steps_per_sec'], alpha=0.8)
        axes[1, 0].set_title('Training Speed')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Steps/Second')
        axes[1, 0].grid(True, alpha=0.3)
    
    # AMP Scale (if using AMP)
    if config.training.use_amp and stats['amp_scales']:
        if stats['amp_scales']['advantage']:
            axes[1, 1].plot(stats['amp_scales']['advantage'], label='Advantage', alpha=0.8)
        if stats['amp_scales']['strategy']:
            axes[1, 1].plot(stats['amp_scales']['strategy'], label='Strategy', alpha=0.8)
        axes[1, 1].set_title('AMP Loss Scaling')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Scale Factor')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'AMP Not Enabled', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('AMP Loss Scaling')
    
    # Training configuration summary
    config_text = f"""Configuration Summary:
    
Model: {config.model.name}
Hidden Size: {config.model.hidden_size}
Dropout: {config.model.dropout}
LayerNorm: {config.model.use_layer_norm}
Residuals: {config.model.use_residuals}

Optimizer: {config.training.optimizer}
Advantage LR: {config.training.advantage_lr}
Strategy LR: {config.training.strategy_lr}
Weight Decay: {config.training.weight_decay}
Scheduler: {config.training.scheduler}

Batch Size: {config.training.batch_size}
Gradient Clip: {config.training.gradient_clip_norm}
AMP: {config.training.use_amp}
Target Norm: {config.training.normalize_targets}

Final Profit: {final_profit:.2f}
"""
    
    axes[1, 2].text(0.05, 0.95, config_text, ha='left', va='top', transform=axes[1, 2].transAxes,
                   fontfamily='monospace', fontsize=8)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    # Save plot
    plot_path = os.path.join(docs_dir, 'training_report.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create markdown report
    report_path = os.path.join(docs_dir, 'training_report.md')
    with open(report_path, 'w') as f:
        f.write(f"# Training Report - {config.model.name}\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Model**: {config.model.name}\n")
        f.write(f"- **Hidden Size**: {config.model.hidden_size}\n")
        f.write(f"- **Dropout**: {config.model.dropout}\n")
        f.write(f"- **LayerNorm**: {config.model.use_layer_norm}\n")
        f.write(f"- **Residuals**: {config.model.use_residuals}\n")
        f.write(f"- **Optimizer**: {config.training.optimizer}\n")
        f.write(f"- **Advantage LR**: {config.training.advantage_lr}\n")
        f.write(f"- **Strategy LR**: {config.training.strategy_lr}\n")
        f.write(f"- **Weight Decay**: {config.training.weight_decay}\n")
        f.write(f"- **Scheduler**: {config.training.scheduler}\n")
        f.write(f"- **Batch Size**: {config.training.batch_size}\n")
        f.write(f"- **Gradient Clip**: {config.training.gradient_clip_norm}\n")
        f.write(f"- **AMP**: {config.training.use_amp}\n")
        f.write(f"- **Target Normalization**: {config.training.normalize_targets}\n\n")
        
        f.write("## Results\n\n")
        f.write(f"- **Final Profit vs Random**: {final_profit:.2f}\n")
        
        if stats['advantage_loss']:
            f.write(f"- **Final Advantage Loss**: {stats['advantage_loss'][-1]:.6f}\n")
        if stats['strategy_loss']:
            f.write(f"- **Final Strategy Loss**: {stats['strategy_loss'][-1]:.6f}\n")
        
        if 'current_lr' in stats:
            f.write(f"- **Current Advantage LR**: {stats['current_lr']['advantage']:.2e}\n")
            f.write(f"- **Current Strategy LR**: {stats['current_lr']['strategy']:.2e}\n")
        
        if stats['steps_per_sec']:
            avg_speed = np.mean(stats['steps_per_sec'][-10:])  # Average of last 10 measurements
            f.write(f"- **Average Training Speed**: {avg_speed:.1f} steps/sec\n")
        
        if 'target_normalization' in stats:
            tn_stats = stats['target_normalization']
            f.write(f"- **Target Normalization Stats**: {tn_stats}\n")
        
        f.write("\n## Training Curves\n\n")
        f.write("![Training Report](training_report.png)\n\n")
        
        f.write("## Notes\n\n")
        f.write("This report was automatically generated by the enhanced training system.\n")
        f.write("The plots show key training metrics over time, including loss curves, learning rates, ")
        f.write("gradient norms, and training speed.\n")
    
    print(f"Training report saved to: {report_path}")
    return report_path


def train_enhanced_deep_cfr(config_path: str, num_iterations: int = 1000, 
                           traversals_per_iteration: int = 200, save_dir: str = "models",
                           log_dir: str = "logs", verbose: bool = False):
    """
    Train an enhanced Deep CFR agent using configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        num_iterations: Number of CFR iterations
        traversals_per_iteration: Number of traversals per iteration
        save_dir: Directory to save models and reports
        log_dir: Directory for tensorboard logs
        verbose: Whether to print verbose output
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration to output directory for reproducibility
    save_config(config, os.path.join(save_dir, "config.yaml"))
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Initialize the enhanced agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    agent = EnhancedDeepCFRAgent(
        config=config,
        player_id=0,
        num_players=6,
        device=device
    )
    
    # Create random opponents
    random_agents = [RandomAgent(i) for i in range(6)]
    
    # Track performance
    profits = []
    
    # Initial evaluation
    print("Initial evaluation...")
    initial_profit = evaluate_against_random(agent, num_games=500, num_players=6)
    profits.append(initial_profit)
    print(f"Initial average profit per game: {initial_profit:.2f}")
    writer.add_scalar('Performance/Profit', initial_profit, 0)
    
    # Checkpoint frequency
    checkpoint_frequency = 100
    
    # Training loop
    print(f"Starting enhanced Deep CFR training for {num_iterations} iterations")
    print(f"Configuration: {config_path}")
    
    for iteration in range(1, num_iterations + 1):
        agent.iteration_count = iteration
        start_time = time.time()
        
        print(f"Iteration {iteration}/{num_iterations}")
        
        # Run traversals to collect data
        print("  Collecting data...")
        for _ in range(traversals_per_iteration):
            state = pkrs.State.from_seed(
                n_players=6,
                button=random.randint(0, 5),
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )
            
            agent.cfr_traverse(state, iteration, random_agents)
        
        # Track traversal time
        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)
        
        # Train advantage network
        print("  Training advantage network...")
        adv_loss = agent.train_advantage_network()
        print(f"  Advantage network loss: {adv_loss:.6f}")
        
        # Log metrics to tensorboard
        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        writer.add_scalar('Memory/Advantage', len(agent.advantage_memory), iteration)
        
        # Log enhanced metrics
        stats = agent.get_training_stats()
        if config.logging.log_lr and 'current_lr' in stats:
            writer.add_scalar('LearningRate/Advantage', stats['current_lr']['advantage'], iteration)
            writer.add_scalar('LearningRate/Strategy', stats['current_lr']['strategy'], iteration)
        
        if config.logging.log_grad_norms and stats['grad_norms']['advantage']:
            writer.add_scalar('GradientNorm/Advantage', stats['grad_norms']['advantage'][-1], iteration)
        
        if config.logging.log_steps_per_sec and stats['steps_per_sec']:
            writer.add_scalar('Speed/StepsPerSec', stats['steps_per_sec'][-1], iteration)
        
        if config.training.use_amp and config.logging.log_amp_scale and stats['amp_scales']:
            if stats['amp_scales']['advantage']:
                writer.add_scalar('AMP/AdvantageScale', stats['amp_scales']['advantage'][-1], iteration)
        
        # Every few iterations, train strategy network and evaluate
        if iteration % 10 == 0 or iteration == num_iterations:
            print("  Training strategy network...")
            strat_loss = agent.train_strategy_network()
            print(f"  Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar('Loss/Strategy', strat_loss, iteration)
            
            if config.logging.log_grad_norms and stats['grad_norms']['strategy']:
                writer.add_scalar('GradientNorm/Strategy', stats['grad_norms']['strategy'][-1], iteration)
            
            if config.training.use_amp and config.logging.log_amp_scale and stats['amp_scales']:
                if stats['amp_scales']['strategy']:
                    writer.add_scalar('AMP/StrategyScale', stats['amp_scales']['strategy'][-1], iteration)
            
            # Evaluate the agent
            print("  Evaluating agent...")
            avg_profit = evaluate_against_random(agent, num_games=500, num_players=6)
            profits.append(avg_profit)
            print(f"  Average profit per game: {avg_profit:.2f}")
            writer.add_scalar('Performance/Profit', avg_profit, iteration)
            
            # Check for early convergence target (≥20% improvement by epoch 5)
            if iteration >= 50 and len(profits) >= 2:  # Roughly epoch 5
                improvement = (avg_profit - initial_profit) / abs(initial_profit) if initial_profit != 0 else 0
                print(f"  Current improvement: {improvement*100:.1f}%")
                writer.add_scalar('Performance/Improvement', improvement*100, iteration)
        
        # Save checkpoint periodically
        if iteration % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f"enhanced_checkpoint_iter_{iteration}.pt")
            agent.save_model(checkpoint_path)
            print(f"  Enhanced checkpoint saved to {checkpoint_path}")
        
        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print(f"  Advantage memory size: {len(agent.advantage_memory)}")
        print(f"  Strategy memory size: {len(agent.strategy_memory)}")
        writer.add_scalar('Memory/Strategy', len(agent.strategy_memory), iteration)
        
        writer.flush()
        print()
    
    # Final evaluation
    print("Final evaluation...")
    final_profit = evaluate_against_random(agent, num_games=1000)
    print(f"Final performance: Average profit per game: {final_profit:.2f}")
    writer.add_scalar('Performance/FinalProfit', final_profit, 0)
    
    # Calculate final improvement
    final_improvement = (final_profit - initial_profit) / abs(initial_profit) if initial_profit != 0 else 0
    print(f"Final improvement: {final_improvement*100:.1f}%")
    writer.add_scalar('Performance/FinalImprovement', final_improvement*100, 0)
    
    # Check convergence gates
    convergence_achieved = final_improvement >= 0.20  # ≥20% improvement
    print(f"Convergence gate (≥20% improvement): {'PASSED' if convergence_achieved else 'FAILED'}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "enhanced_final_model.pt")
    agent.save_model(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Generate training report
    report_path = create_training_report(agent, config, save_dir, final_profit)
    
    writer.close()
    
    return agent, profits, convergence_achieved


def create_ablation_script():
    """Create a script to run all ablation studies."""
    script_content = """#!/bin/bash
# Ablation Study Script for Neural Network Convergence Accelerator

echo "Starting Neural Network Convergence Accelerator Ablation Studies"
echo "================================================================"

# Create results directory
mkdir -p results/ablation

# Baseline configuration
echo "Running baseline configuration..."
python src/training/train_enhanced.py \\
    --config configs/ablate/baseline.yaml \\
    --iterations 1000 \\
    --save-dir results/ablation/baseline \\
    --log-dir logs/ablation/baseline \\
    --verbose

# AdamW + Clipping configuration
echo "Running AdamW + Clipping configuration..."
python src/training/train_enhanced.py \\
    --config configs/ablate/adamw_clip.yaml \\
    --iterations 1000 \\
    --save-dir results/ablation/adamw_clip \\
    --log-dir logs/ablation/adamw_clip \\
    --verbose

# Cosine + AMP configuration
echo "Running Cosine + AMP configuration..."
python src/training/train_enhanced.py \\
    --config configs/ablate/cosine_amp.yaml \\
    --iterations 1000 \\
    --save-dir results/ablation/cosine_amp \\
    --log-dir logs/ablation/cosine_amp \\
    --verbose

# Target Normalization configuration
echo "Running Target Normalization configuration..."
python src/training/train_enhanced.py \\
    --config configs/ablate/norm_targets.yaml \\
    --iterations 1000 \\
    --save-dir results/ablation/norm_targets \\
    --log-dir logs/ablation/norm_targets \\
    --verbose

echo "All ablation studies completed!"
echo "View results with: tensorboard --logdir logs/ablation"
echo "Training reports are available in results/ablation/*/docs/"
"""
    
    with open("run_ablation_studies.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("run_ablation_studies.sh", 0o755)
    print("Ablation study script created: run_ablation_studies.sh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Enhanced Deep CFR agent')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--strict', action='store_true', help='Enable strict error checking')
    parser.add_argument('--create-ablation', action='store_true', help='Create ablation study script')
    
    args = parser.parse_args()
    
    if args.create_ablation:
        create_ablation_script()
        exit(0)
    
    # Set strict checking
    set_strict_checking(args.strict)
    
    try:
        agent, profits, convergence_achieved = train_enhanced_deep_cfr(
            config_path=args.config,
            num_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            verbose=args.verbose
        )
        
        print("\nTraining Summary:")
        print(f"Configuration: {args.config}")
        print(f"Convergence gate: {'PASSED' if convergence_achieved else 'FAILED'}")
        if profits:
            initial_profit = profits[0]
            final_profit = profits[-1]
            improvement = (final_profit - initial_profit) / abs(initial_profit) * 100 if initial_profit != 0 else 0
            print(f"Initial profit: {initial_profit:.2f}")
            print(f"Final profit: {final_profit:.2f}")
            print(f"Improvement: {improvement:.1f}%")
        
        print(f"\nView training progress:")
        print(f"tensorboard --logdir {args.log_dir}")
        print(f"Training report: {args.save_dir}/docs/training_report.md")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise