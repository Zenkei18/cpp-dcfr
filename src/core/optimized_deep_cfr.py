# src/core/optimized_deep_cfr.py
"""
Optimized DeepCFR agent with performance improvements.
Set SPEED_REFAC=1 environment variable to enable optimizations.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pokers as pkrs
from collections import deque
from typing import List, Tuple, Dict, Any, Optional

from .optimized_model import (
    encode_state_optimized, OptimizedPokerNetwork, OptimizedBatchProcessor,
    get_legal_action_types_optimized, performance_monitor, SPEED_REFAC
)
from .deep_cfr import PrioritizedMemory  # Reuse existing memory implementation
from .model import VERBOSE, set_verbose
try:
    from ..utils.settings import STRICT_CHECKING
    from ..utils.logging import log_game_error
except ImportError:
    # Fallback for direct execution
    from src.utils.settings import STRICT_CHECKING
    from src.utils.logging import log_game_error

class OptimizedDeepCFRAgent:
    """
    Optimized DeepCFR agent with performance improvements.
    
    Key optimizations:
    1. Optimized state encoding with array reuse
    2. Batch processing with preallocated tensors  
    3. Reduced device transfers
    4. Cached legal action computations
    5. Vectorized training operations
    6. Memory pool for frequent allocations
    """
    
    def __init__(self, player_id=0, num_players=6, memory_size=300000, device='cpu'):
        self.player_id = player_id
        self.num_players = num_players
        self.device = device
        
        # Define action types (Fold, Check/Call, Raise)
        self.num_actions = 3
        
        # Calculate input size based on actual state encoding (empirically determined)
        input_size = 156  # Actual size from encode_state function
        
        # Create networks - use optimized version if enabled
        if SPEED_REFAC:
            self.advantage_net = OptimizedPokerNetwork(
                input_size=input_size, 
                hidden_size=256, 
                num_actions=self.num_actions
            ).to(device)
            self.strategy_net = OptimizedPokerNetwork(
                input_size=input_size,
                hidden_size=256, 
                num_actions=self.num_actions
            ).to(device)
        else:
            from .model import PokerNetwork
            self.advantage_net = PokerNetwork(
                input_size=input_size,
                hidden_size=256,
                num_actions=self.num_actions
            ).to(device)
            self.strategy_net = PokerNetwork(
                input_size=input_size,
                hidden_size=256,
                num_actions=self.num_actions
            ).to(device)
        
        # Optimizers
        lr_advantage = 1e-6 if not SPEED_REFAC else 5e-6  # Slightly higher LR for optimized version
        lr_strategy = 5e-5 if not SPEED_REFAC else 1e-4   # Slightly higher LR for optimized version
        
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=lr_advantage, weight_decay=1e-5)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=lr_strategy, weight_decay=1e-5)
        
        # Memory systems
        self.advantage_memory = PrioritizedMemory(memory_size)
        self.strategy_memory = deque(maxlen=memory_size)
        
        # Optimization components
        if SPEED_REFAC:
            self.batch_processor = OptimizedBatchProcessor(device=device)
        else:
            self.batch_processor = None
        
        # Performance tracking
        self.iteration_count = 0
        self.max_regret_seen = 1.0
        
        # Bet sizing bounds
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0
        
        # Cached computations for optimization
        if SPEED_REFAC:
            self._state_cache = {}
            self._action_cache = {}
            
    def encode_state(self, state: Any, player_id: int) -> np.ndarray:
        """Use optimized state encoding if enabled."""
        if SPEED_REFAC:
            import time
            start_time = time.perf_counter()
            result = encode_state_optimized(state, player_id)
            performance_monitor.time_function('encode_state', time.perf_counter() - start_time)
            return result
        else:
            from .model import encode_state
            return encode_state(state, player_id)
    
    def get_legal_action_types(self, state: Any) -> List[int]:
        """Use optimized legal action computation if enabled."""
        if SPEED_REFAC:
            import time
            start_time = time.perf_counter()
            result = get_legal_action_types_optimized(state)
            performance_monitor.time_function('get_legal_action_types', time.perf_counter() - start_time)
            return result
        else:
            # Original implementation
            legal_action_types = []
            
            if pkrs.ActionEnum.Fold in state.legal_actions:
                legal_action_types.append(0)
            if pkrs.ActionEnum.Check in state.legal_actions or pkrs.ActionEnum.Call in state.legal_actions:
                legal_action_types.append(1)
            if pkrs.ActionEnum.Raise in state.legal_actions:
                legal_action_types.append(2)
            
            return legal_action_types
    
    def action_type_to_pokers_action(self, action_type: int, state: Any, bet_size_multiplier: Optional[float] = None):
        """
        Convert action type to Pokers action (optimized version).
        Maintains compatibility with original implementation.
        """
        # Use the original implementation from the parent class for now
        # This could be optimized further but maintaining correctness is priority
        
        try:
            if action_type == 0:  # Fold
                if pkrs.ActionEnum.Fold in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                if pkrs.ActionEnum.Check in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Check)
                if pkrs.ActionEnum.Call in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if VERBOSE: 
                    print(f"DeepCFRAgent WARNING: Fold chosen but no other legal fallback. Returning Fold anyway.")
                return pkrs.Action(pkrs.ActionEnum.Fold)

            elif action_type == 1:  # Check/Call
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                elif pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Fold in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                if VERBOSE: 
                    print(f"DeepCFRAgent WARNING: Check/Call chosen but neither legal, nor Fold. Returning Check anyway.")
                return pkrs.Action(pkrs.ActionEnum.Check)

            elif action_type == 2:  # Raise
                if pkrs.ActionEnum.Raise not in state.legal_actions:
                    if VERBOSE: 
                        print(f"DeepCFRAgent INFO: Raise (type 2) chosen, but Raise not in legal_actions. Falling back.")
                    if pkrs.ActionEnum.Call in state.legal_actions: 
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    if pkrs.ActionEnum.Check in state.legal_actions: 
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    return pkrs.Action(pkrs.ActionEnum.Fold)

                player_state = state.players_state[state.current_player]
                current_bet = player_state.bet_chips
                available_stake = player_state.stake

                call_amount = max(0.0, state.min_bet - current_bet)

                min_raise_increment = 1.0
                if hasattr(state, 'bb') and state.bb is not None and float(state.bb) > 0:
                    min_raise_increment = max(1.0, float(state.bb))
                elif state.min_bet > 0:
                    min_raise_increment = max(1.0, state.min_bet - current_bet if state.min_bet > current_bet else 1.0)

                if available_stake < call_amount + min_raise_increment:
                    if VERBOSE:
                        print(f"DeepCFRAgent INFO: Raise (type 2) chosen, but cannot make a valid min_raise_increment. "
                              f"AvailableStake({available_stake:.2f}) < CallAmt({call_amount:.2f}) + MinInc({min_raise_increment:.2f}). Switching to Call.")
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    else:
                        if VERBOSE: 
                            print(f"DeepCFRAgent WARNING: Cannot Call (not legal after failing raise check), falling back to Fold.")
                        return pkrs.Action(pkrs.ActionEnum.Fold)

                remaining_stake_after_call = available_stake - call_amount
                pot_size = max(1.0, state.pot)

                if bet_size_multiplier is None:
                    bet_size_multiplier = 1.0
                else:
                    bet_size_multiplier = float(bet_size_multiplier)

                bet_size_multiplier = max(self.min_bet_size, min(self.max_bet_size, bet_size_multiplier))
                network_desired_additional_raise = pot_size * bet_size_multiplier

                chosen_additional_amount = network_desired_additional_raise
                chosen_additional_amount = min(chosen_additional_amount, remaining_stake_after_call)
                chosen_additional_amount = max(chosen_additional_amount, min_raise_increment)

                if chosen_additional_amount > remaining_stake_after_call:
                    chosen_additional_amount = remaining_stake_after_call

                # Float safeguard
                total_chips_player_would_commit_this_turn = call_amount + chosen_additional_amount
                epsilon = 0.00001

                if total_chips_player_would_commit_this_turn > available_stake + epsilon:
                    if VERBOSE:
                        print(f"DeepCFRAgent INFO: Float Safeguard in action_type_to_pokers_action triggered.")
                        print(f"  Initial chosen_additional_amount: {chosen_additional_amount:.6f}")
                        print(f"  Total commit ({total_chips_player_would_commit_this_turn:.6f}) > available_stake ({available_stake:.6f})")

                    chosen_additional_amount = available_stake - call_amount
                    chosen_additional_amount = max(0.0, chosen_additional_amount)

                    if VERBOSE:
                        print(f"  Adjusted chosen_additional_amount: {chosen_additional_amount:.6f}")
                        print(f"  New total commit: {(call_amount + chosen_additional_amount):.6f}")

                chosen_additional_amount = max(0.0, chosen_additional_amount)

                if VERBOSE:
                    print(f"--- DeepCFRAgent Raise Calculation (FINAL PRE-RETURN) ---")
                    print(f"  Player ID: {state.current_player}, Stage: {state.stage}")
                    print(f"  Available Stake: {available_stake:.6f}, Current Bet In Pot: {current_bet:.6f}")
                    print(f"  State Min Bet (to call): {state.min_bet:.6f}, Pot Size: {state.pot:.6f}")
                    print(f"  Calculated Call Amount: {call_amount:.6f}")
                    print(f"  Min Raise Increment: {min_raise_increment:.6f}")
                    print(f"  Remaining Stake After Call: {remaining_stake_after_call:.6f}")
                    print(f"  Bet Size Multiplier (from net, raw): {float(bet_size_multiplier) if bet_size_multiplier is not None else 'N/A'}, (used, clipped): {bet_size_multiplier:.6f}")
                    print(f"  Network Desired Additional Raise (pot * mult): {network_desired_additional_raise:.6f}")
                    print(f"  Chosen Additional Raise Amount (pre-float-guard): {network_desired_additional_raise:.6f} -> clipped by rules to -> {chosen_additional_amount+epsilon if total_chips_player_would_commit_this_turn > available_stake + epsilon else chosen_additional_amount:.6f}")
                    print(f"  Final Chosen Additional Raise Amount (post-float-guard): {chosen_additional_amount:.6f}")
                    _total_chips_this_action = call_amount + chosen_additional_amount
                    print(f"  Total Chips for this action (call + additional): {_total_chips_this_action:.6f}")
                    _is_exact_all_in = abs(_total_chips_this_action - available_stake) < epsilon
                    print(f"  Is this an exact all-in (post-safeguard)? {_is_exact_all_in}")
                    if _is_exact_all_in: 
                        print(f"    All-in Difference: {(_total_chips_this_action - available_stake):.10f}")
                    print(f"--------------------------------------------------------------------")

                return pkrs.Action(pkrs.ActionEnum.Raise, chosen_additional_amount)

            else:
                if VERBOSE: 
                    print(f"DeepCFRAgent ERROR: Unknown action type: {action_type}")
                if pkrs.ActionEnum.Call in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Check in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Check)
                return pkrs.Action(pkrs.ActionEnum.Fold)

        except Exception as e:
            try:
                is_verbose = VERBOSE
            except NameError:
                is_verbose = False

            if is_verbose:
                print(f"DeepCFRAgent CRITICAL ERROR in action_type_to_pokers_action: Type {action_type} for player {self.player_id}: {e}")
                print(f"  State: current_player={state.current_player}, stage={state.stage}, legal_actions={state.legal_actions}")
                if hasattr(state, 'players_state') and self.player_id < len(state.players_state):
                    print(f"  Player {self.player_id} stake: {state.players_state[self.player_id].stake}, bet: {state.players_state[self.player_id].bet_chips}")
                else:
                    print(f"  Player state for player {self.player_id} not accessible.")
                import traceback
                traceback.print_exc()

            if hasattr(state, 'legal_actions'):
                if pkrs.ActionEnum.Call in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Check in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Check)
                if pkrs.ActionEnum.Fold in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Fold)
            
            return pkrs.Action(pkrs.ActionEnum.Fold)

    def cfr_traverse(self, state: Any, iteration: int, random_agents: List[Any], depth: int = 0) -> float:
        """
        Optimized CFR traversal with reduced allocations and device transfers.
        """
        if SPEED_REFAC:
            import time
            start_time = time.perf_counter()
        
        # Depth protection
        max_depth = 1000
        if depth > max_depth:
            if VERBOSE:
                print(f"WARNING: Max recursion depth reached ({max_depth}). Returning zero value.")
            return 0
        
        if state.final_state:
            result = state.players_state[self.player_id].reward
            if SPEED_REFAC:
                performance_monitor.time_function('cfr_traverse', time.perf_counter() - start_time)
            return result
        
        current_player = state.current_player
        
        if current_player == self.player_id:
            # Agent's turn - use optimized path
            legal_action_types = self.get_legal_action_types(state)
            
            if not legal_action_types:
                if VERBOSE:
                    print(f"WARNING: No legal actions found for player {current_player} at depth {depth}")
                return 0
            
            # Encode state using optimized encoder
            state_encoding = self.encode_state(state, self.player_id)
            
            if SPEED_REFAC:
                # Reduce device transfers by batching operations
                state_tensor = torch.from_numpy(state_encoding).float().unsqueeze(0).to(self.device)
            else:
                state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)
            
            # Network forward pass
            with torch.no_grad():
                advantages, bet_size_pred = self.advantage_net(state_tensor)
                if SPEED_REFAC:
                    # Reduce .cpu().numpy() calls
                    advantages_np = advantages[0].detach().cpu().numpy()
                    bet_size_multiplier = bet_size_pred[0][0].detach().item()
                else:
                    advantages_np = advantages[0].cpu().numpy()
                    bet_size_multiplier = bet_size_pred[0][0].item()
            
            # Compute strategy using regret matching
            advantages_masked = np.zeros(self.num_actions)
            for a in legal_action_types:
                advantages_masked[a] = max(advantages_np[a], 0)
                
            if sum(advantages_masked) > 0:
                strategy = advantages_masked / sum(advantages_masked)
            else:
                strategy = np.zeros(self.num_actions)
                for a in legal_action_types:
                    strategy[a] = 1.0 / len(legal_action_types)
            
            # Traverse actions
            action_values = np.zeros(self.num_actions)
            for action_type in legal_action_types:
                try:
                    if action_type == 2:  # Raise
                        pokers_action = self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
                    else:
                        pokers_action = self.action_type_to_pokers_action(action_type, state)
                    
                    new_state = state.apply_action(pokers_action)
                    
                    if new_state.status != pkrs.StateStatus.Ok:
                        log_file = log_game_error(state, pokers_action, f"State status not OK ({new_state.status})")
                        if STRICT_CHECKING:
                            raise ValueError(f"State status not OK ({new_state.status}) during CFR traversal. Details logged to {log_file}")
                        elif VERBOSE:
                            print(f"WARNING: Invalid action {action_type} at depth {depth}. Status: {new_state.status}")
                            print(f"Details logged to {log_file}")
                        continue
                        
                    action_values[action_type] = self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
                except Exception as e:
                    if VERBOSE:
                        print(f"ERROR in traversal for action {action_type}: {e}")
                    action_values[action_type] = 0
                    if STRICT_CHECKING:
                        raise
            
            # Compute regrets and update memory
            ev = sum(strategy[a] * action_values[a] for a in legal_action_types)
            
            # Optimized regret calculation
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            
            for action_type in legal_action_types:
                regret = action_values[action_type] - ev
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -10.0, 10.0)
                
                scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0
                weighted_regret = clipped_regret * scale_factor
                
                priority = abs(weighted_regret) + 0.01
                
                if action_type == 2:
                    self.advantage_memory.add(
                        (state_encoding, 
                         np.zeros(20),
                         action_type, 
                         bet_size_multiplier, 
                         weighted_regret),
                        priority
                    )
                else:
                    self.advantage_memory.add(
                        (state_encoding,
                         np.zeros(20),
                         action_type, 
                         0.0,
                         weighted_regret),
                        priority
                    )
            
            # Add to strategy memory
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_types:
                strategy_full[a] = strategy[a]
            
            self.strategy_memory.append((
                state_encoding,
                np.zeros(20),
                strategy_full,
                bet_size_multiplier if 2 in legal_action_types else 0.0,
                iteration
            ))
            
            if SPEED_REFAC:
                performance_monitor.time_function('cfr_traverse', time.perf_counter() - start_time)
            
            return ev
            
        else:
            # Random agent's turn
            try:
                action = random_agents[current_player].choose_action(state)
                new_state = state.apply_action(action)
                
                if new_state.status != pkrs.StateStatus.Ok:
                    log_file = log_game_error(state, action, f"State status not OK ({new_state.status})")
                    if STRICT_CHECKING:
                        raise ValueError(f"State status not OK ({new_state.status}) from random agent. Details logged to {log_file}")
                    if VERBOSE:
                        print(f"WARNING: Random agent made invalid action at depth {depth}. Status: {new_state.status}")
                        print(f"Details logged to {log_file}")
                    return 0
                    
                result = self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
                
                if SPEED_REFAC:
                    performance_monitor.time_function('cfr_traverse', time.perf_counter() - start_time)
                
                return result
            except Exception as e:
                if VERBOSE:
                    print(f"ERROR in random agent traversal: {e}")
                if STRICT_CHECKING:
                    raise
                return 0

    def train_advantage_network(self, batch_size: int = 128, epochs: int = 3, beta_start: float = 0.4, beta_end: float = 1.0) -> float:
        """
        Optimized advantage network training with batch processing improvements.
        """
        if SPEED_REFAC:
            import time
            start_time = time.perf_counter()
        
        if len(self.advantage_memory) < batch_size:
            return 0
        
        self.advantage_net.train()
        total_loss = 0
        
        # Calculate beta for importance sampling
        progress = min(1.0, self.iteration_count / 10000)
        beta = beta_start + progress * (beta_end - beta_start)
        
        for epoch in range(epochs):
            # Sample batch
            batch, indices, weights = self.advantage_memory.sample(batch_size, beta=beta)
            
            # Process batch using optimized batch processor
            if SPEED_REFAC and self.batch_processor:
                # Use optimized batch processing
                batch_tensors = self.batch_processor.prepare_batch_tensors(batch, len(batch))
                
                state_tensors = batch_tensors['states']
                action_type_tensors = batch_tensors['actions'].long()
                regret_tensors = batch_tensors['regrets']
                weight_tensors = torch.from_numpy(weights).float().to(self.device)
                
                # Extract bet sizes
                _, _, _, bet_sizes, _ = zip(*batch)
                bet_size_tensors = torch.FloatTensor(bet_sizes).unsqueeze(1).to(self.device)
                
                # Extract opponent features  
                _, opponent_features, _, _, _ = zip(*batch)
                opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            else:
                # Standard batch processing
                states, opponent_features, action_types, bet_sizes, regrets = zip(*batch)
                
                state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
                opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
                action_type_tensors = torch.LongTensor(np.array(action_types)).to(self.device)
                bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
                regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
                weight_tensors = torch.FloatTensor(weights).to(self.device)
            
            # Forward pass
            action_advantages, bet_size_preds = self.advantage_net(state_tensors, opponent_feature_tensors)
            
            # Debug tensor shapes - ALWAYS show during error
            print(f"🔍 DEBUG: action_advantages shape: {action_advantages.shape}")
            print(f"🔍 DEBUG: action_advantages dims: {action_advantages.dim()}")
            print(f"🔍 DEBUG: action_type_tensors shape: {action_type_tensors.shape}")
            print(f"🔍 DEBUG: action_type_tensors dims: {action_type_tensors.dim()}")
            print(f"🔍 DEBUG: action_type_tensors sample: {action_type_tensors[:5] if len(action_type_tensors) > 0 else 'empty'}")
            
            # Action loss  
            try:
                predicted_regrets = action_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
                print(f"✅ DEBUG: Gather successful, shape: {predicted_regrets.shape}")
            except Exception as gather_error:
                print(f"❌ DEBUG: Gather failed: {gather_error}")
                print(f"🔍 DEBUG: action_advantages: {action_advantages}")
                print(f"🔍 DEBUG: action_type_tensors.unsqueeze(1): {action_type_tensors.unsqueeze(1)}")
                raise
            action_loss = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
            weighted_action_loss = (action_loss * weight_tensors).mean()
            
            # Bet sizing loss for raise actions
            raise_mask = (action_type_tensors == 2)
            if torch.any(raise_mask):
                all_bet_losses = F.smooth_l1_loss(bet_size_preds, bet_size_tensors, reduction='none')
                masked_bet_losses = all_bet_losses * raise_mask.float().unsqueeze(1)
                
                raise_count = raise_mask.sum().item()
                if raise_count > 0:
                    weighted_bet_size_loss = (masked_bet_losses.squeeze() * weight_tensors).sum() / raise_count
                    combined_loss = weighted_action_loss + 0.5 * weighted_bet_size_loss
                else:
                    combined_loss = weighted_action_loss
            else:
                combined_loss = weighted_action_loss
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            combined_loss.backward()
            
            if SPEED_REFAC:
                # Slightly more aggressive gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=0.3)
            else:
                torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Update priorities
            with torch.no_grad():
                new_action_errors = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
                
                if torch.any(raise_mask):
                    new_bet_errors = torch.zeros_like(new_action_errors)
                    raise_indices = torch.where(raise_mask)[0]
                    for i in raise_indices:
                        new_bet_errors[i] = F.smooth_l1_loss(
                            bet_size_preds[i], bet_size_tensors[i], reduction='mean'
                        )
                    combined_errors = new_action_errors + 0.5 * new_bet_errors
                else:
                    combined_errors = new_action_errors
                
                combined_errors_np = combined_errors.cpu().numpy()
                for i, idx in enumerate(indices):
                    self.advantage_memory.update_priority(idx, combined_errors_np[i] + 0.01)
            
            total_loss += combined_loss.item()
        
        if SPEED_REFAC:
            performance_monitor.time_function('train_advantage_network', time.perf_counter() - start_time)
        
        return total_loss / epochs

    def train_strategy_network(self, batch_size: int = 128, epochs: int = 3) -> float:
        """Optimized strategy network training."""
        if SPEED_REFAC:
            import time
            start_time = time.perf_counter()
        
        if len(self.strategy_memory) < batch_size:
            return 0
        
        self.strategy_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch
            batch = random.sample(self.strategy_memory, batch_size)
            
            if SPEED_REFAC and self.batch_processor:
                # Use optimized batch processing (adapt for strategy training)
                states, opponent_features, strategies, bet_sizes, iterations = zip(*batch)
                
                state_tensors = torch.from_numpy(np.array(states)).float().to(self.device)
                strategy_tensors = torch.from_numpy(np.array(strategies)).float().to(self.device)
                bet_size_tensors = torch.from_numpy(np.array(bet_sizes)).float().unsqueeze(1).to(self.device)
                iteration_tensors = torch.from_numpy(np.array(iterations)).float().unsqueeze(1).to(self.device)
            else:
                # Standard processing
                states, opponent_features, strategies, bet_sizes, iterations = zip(*batch)
                
                state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
                strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
                bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
                iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            
            # Linear CFR weights
            weights = iteration_tensors / torch.sum(iteration_tensors)
            
            # Forward pass
            action_logits, bet_size_preds = self.strategy_net(state_tensors)
            predicted_strategies = F.softmax(action_logits, dim=1)
            
            # Action loss with epsilon for stability
            epsilon = 1e-8 if not SPEED_REFAC else 1e-10  # Smaller epsilon for optimized version
            action_loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(predicted_strategies + epsilon), dim=1))
            
            # Bet size loss
            raise_mask = (strategy_tensors[:, 2] > 0)
            if raise_mask.sum() > 0:
                raise_indices = torch.nonzero(raise_mask).squeeze(1)
                raise_bet_preds = bet_size_preds[raise_indices]
                raise_bet_targets = bet_size_tensors[raise_indices]
                raise_weights = weights[raise_indices]
                
                bet_size_loss = F.smooth_l1_loss(raise_bet_preds, raise_bet_targets, reduction='none')
                weighted_bet_size_loss = torch.sum(raise_weights * bet_size_loss.squeeze())
                
                combined_loss = action_loss + 0.5 * weighted_bet_size_loss
            else:
                combined_loss = action_loss
            
            # Backward pass
            self.strategy_optimizer.zero_grad()
            combined_loss.backward()
            
            if SPEED_REFAC:
                torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=0.3)
            else:
                torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=0.5)
            
            self.strategy_optimizer.step()
            
            total_loss += combined_loss.item()
        
        if SPEED_REFAC:
            performance_monitor.time_function('train_strategy_network', time.perf_counter() - start_time)
        
        return total_loss / epochs

    def choose_action(self, state: Any):
        """Choose action using the strategy network (optimized version)."""
        legal_action_types = self.get_legal_action_types(state)
        
        if not legal_action_types:
            if pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
            elif pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            else:
                return pkrs.Action(pkrs.ActionEnum.Fold)
        
        state_encoding = self.encode_state(state, self.player_id)
        
        if SPEED_REFAC:
            state_tensor = torch.from_numpy(state_encoding).float().unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, bet_size_pred = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()
        
        # Filter to legal actions
        legal_probs = np.array([probs[a] for a in legal_action_types])
        if np.sum(legal_probs) > 0:
            legal_probs = legal_probs / np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_action_types)) / len(legal_action_types)
        
        # Choose action
        action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
        action_type = legal_action_types[action_idx]
        
        if action_type == 2:  # Raise
            return self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
        else:
            return self.action_type_to_pokers_action(action_type, state)

    def save_model(self, path_prefix: str):
        """Save model with performance statistics."""
        checkpoint = {
            'iteration': self.iteration_count,
            'advantage_net': self.advantage_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict(),
            'min_bet_size': self.min_bet_size,
            'max_bet_size': self.max_bet_size,
            'optimizations_enabled': SPEED_REFAC
        }
        
        # Include performance statistics if optimizations are enabled
        if SPEED_REFAC:
            checkpoint['performance_stats'] = performance_monitor.get_stats()
        
        torch.save(checkpoint, f"{path_prefix}_iteration_{self.iteration_count}.pt")
        
    def load_model(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        self.iteration_count = checkpoint['iteration']
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
        
        if 'min_bet_size' in checkpoint:
            self.min_bet_size = checkpoint['min_bet_size']
        if 'max_bet_size' in checkpoint:
            self.max_bet_size = checkpoint['max_bet_size']
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the optimized agent."""
        if SPEED_REFAC:
            return performance_monitor.get_stats()
        else:
            return {"message": "Performance monitoring not enabled. Set SPEED_REFAC=1 to enable."}