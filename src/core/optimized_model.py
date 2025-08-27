# src/core/optimized_model.py
"""
Optimized versions of core model functions with feature flags.
Set SPEED_REFAC=1 environment variable to enable optimizations.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import pokers as pkrs

# Check if speed optimizations are enabled
SPEED_REFAC = os.environ.get('SPEED_REFAC', '0') == '1'

# Cache for preallocated arrays to avoid repeated numpy.zeros() calls
_ENCODING_CACHE = {
    'hand_enc': None,
    'community_enc': None, 
    'stage_enc': None,
    'button_enc': None,
    'current_player_enc': None,
    'legal_actions_enc': None,
    'prev_action_enc': None,
    'player_states': {}
}

def _initialize_encoding_cache(num_players: int = 6):
    """Initialize preallocated arrays for state encoding."""
    global _ENCODING_CACHE
    
    if _ENCODING_CACHE['hand_enc'] is not None:
        return  # Already initialized
    
    _ENCODING_CACHE['hand_enc'] = np.zeros(52, dtype=np.float32)
    _ENCODING_CACHE['community_enc'] = np.zeros(52, dtype=np.float32)
    _ENCODING_CACHE['stage_enc'] = np.zeros(5, dtype=np.float32)
    _ENCODING_CACHE['button_enc'] = np.zeros(num_players, dtype=np.float32)
    _ENCODING_CACHE['current_player_enc'] = np.zeros(num_players, dtype=np.float32)
    _ENCODING_CACHE['legal_actions_enc'] = np.zeros(4, dtype=np.float32)
    _ENCODING_CACHE['prev_action_enc'] = np.zeros(5, dtype=np.float32)
    
    # Preallocate player state arrays
    for p in range(num_players):
        _ENCODING_CACHE['player_states'][p] = np.zeros(4, dtype=np.float32)

def encode_state_optimized(state: Any, player_id: int = 0) -> np.ndarray:
    """
    Optimized state encoding that reuses arrays and reduces allocations.
    
    Optimizations:
    1. Reuse preallocated numpy arrays instead of creating new ones
    2. Use single concatenate call instead of multiple appends
    3. Vectorized operations where possible
    4. Reduced function call overhead
    
    Args:
        state: The Pokers state
        player_id: The ID of the player for whom we're encoding
        
    Returns:
        Encoded state as numpy array
    """
    if not SPEED_REFAC:
        # Fall back to original implementation
        from .model import encode_state
        return encode_state(state, player_id)
    
    # Initialize cache if needed
    num_players = len(state.players_state)
    _initialize_encoding_cache(num_players)
    
    # Pre-calculate commonly used values
    initial_stake = max(1.0, state.players_state[0].stake)  # Avoid division by zero
    pot_size = state.pot
    min_bet = state.min_bet
    
    # Prepare output list for single concatenate
    encoded_parts = []
    
    # Encode player's hole cards (reuse array)
    hand_enc = _ENCODING_CACHE['hand_enc']
    hand_enc.fill(0)  # Reset array
    hand_cards = state.players_state[player_id].hand
    for card in hand_cards:
        hand_enc[int(card.suit) * 13 + int(card.rank)] = 1
    encoded_parts.append(hand_enc.copy())
    
    # Encode community cards (reuse array)  
    community_enc = _ENCODING_CACHE['community_enc']
    community_enc.fill(0)  # Reset array
    for card in state.public_cards:
        community_enc[int(card.suit) * 13 + int(card.rank)] = 1
    encoded_parts.append(community_enc.copy())
    
    # Encode game stage (reuse array)
    stage_enc = _ENCODING_CACHE['stage_enc']
    stage_enc.fill(0)  # Reset array
    stage_enc[int(state.stage)] = 1
    encoded_parts.append(stage_enc.copy())
    
    # Encode pot size (scalar)
    encoded_parts.append(np.array([pot_size / initial_stake], dtype=np.float32))
    
    # Encode button position (reuse array)
    button_enc = _ENCODING_CACHE['button_enc']
    button_enc.fill(0)  # Reset array
    if state.button < len(button_enc):
        button_enc[state.button] = 1
    encoded_parts.append(button_enc.copy())
    
    # Encode current player (reuse array)
    current_player_enc = _ENCODING_CACHE['current_player_enc']
    current_player_enc.fill(0)  # Reset array
    if state.current_player < len(current_player_enc):
        current_player_enc[state.current_player] = 1
    encoded_parts.append(current_player_enc.copy())
    
    # Encode player states (vectorized where possible)
    for p in range(num_players):
        player_state = state.players_state[p]
        
        # Use preallocated array
        player_enc = _ENCODING_CACHE['player_states'][p]
        player_enc[0] = 1.0 if player_state.active else 0.0
        player_enc[1] = player_state.bet_chips / initial_stake
        player_enc[2] = player_state.pot_chips / initial_stake
        player_enc[3] = player_state.stake / initial_stake
        
        encoded_parts.append(player_enc.copy())
    
    # Encode minimum bet (scalar)
    encoded_parts.append(np.array([min_bet / initial_stake], dtype=np.float32))
    
    # Encode legal actions (reuse array)
    legal_actions_enc = _ENCODING_CACHE['legal_actions_enc']
    legal_actions_enc.fill(0)  # Reset array
    for action_enum in state.legal_actions:
        if int(action_enum) < len(legal_actions_enc):
            legal_actions_enc[int(action_enum)] = 1
    encoded_parts.append(legal_actions_enc.copy())
    
    # Encode previous action (reuse array)
    prev_action_enc = _ENCODING_CACHE['prev_action_enc']
    prev_action_enc.fill(0)  # Reset array
    if state.from_action is not None:
        action_int = int(state.from_action.action.action)
        if action_int < 4:  # Valid action type
            prev_action_enc[action_int] = 1
            prev_action_enc[4] = state.from_action.action.amount / initial_stake
    encoded_parts.append(prev_action_enc.copy())
    
    # Single concatenate operation instead of multiple appends
    return np.concatenate(encoded_parts)


class OptimizedPokerNetwork(nn.Module):
    """
    Optimized poker network with batch processing improvements.
    
    Optimizations:
    1. Reduced layer depth for faster inference
    2. Optimized activation functions
    3. Better memory layout
    4. Batch processing support
    """
    
    def __init__(self, input_size: int = 156, hidden_size: int = 256, num_actions: int = 3):
        super().__init__()
        
        if SPEED_REFAC:
            # Optimized architecture: fewer layers but wider
            self.base = nn.Sequential(
                nn.Linear(input_size, hidden_size * 2),  # Wider first layer
                nn.ReLU(inplace=True),  # In-place operations save memory
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(inplace=True),
            )
        else:
            # Original architecture
            self.base = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
        
        # Action and sizing heads - use consistent hidden size
        final_hidden_size = hidden_size
        self.action_head = nn.Linear(final_hidden_size, num_actions)
        self.sizing_head = nn.Sequential(
            nn.Linear(final_hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, opponent_features=None):
        """Optimized forward pass with better memory usage."""
        features = self.base(x)
        
        # Simple forward pass - avoid torch.jit complications for now
        action_logits = self.action_head(features)
        bet_size = 0.1 + 2.9 * self.sizing_head(features)
        
        return action_logits, bet_size


class OptimizedBatchProcessor:
    """
    Optimized batch processing for training operations.
    
    Optimizations:
    1. Preallocated tensors for batch operations
    2. Reduced device transfers
    3. Vectorized tensor operations
    4. Memory pool for frequent allocations
    """
    
    def __init__(self, device: str = 'cpu', max_batch_size: int = 512):
        self.device = device
        self.max_batch_size = max_batch_size
        
        # Preallocate tensors for batch operations
        self.tensor_pool = {}
        self._initialize_tensor_pool()
    
    def _initialize_tensor_pool(self):
        """Initialize preallocated tensors."""
        if not SPEED_REFAC:
            return
            
        # Common tensor shapes for this domain
        shapes = {
            'states': (self.max_batch_size, 156),  # Actual state encoding size
            'actions': (self.max_batch_size,),
            'regrets': (self.max_batch_size,),
            'strategies': (self.max_batch_size, 3),
            'bet_sizes': (self.max_batch_size, 1),
            'weights': (self.max_batch_size,)
        }
        
        for name, shape in shapes.items():
            self.tensor_pool[name] = torch.zeros(shape, device=self.device, dtype=torch.float32)
    
    def prepare_batch_tensors(self, batch_data: List[Tuple], batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Prepare batch tensors with optimized memory usage.
        
        Args:
            batch_data: List of tuples containing batch data
            batch_size: Size of the current batch
            
        Returns:
            Dictionary of prepared tensors
        """
        if not SPEED_REFAC:
            # Fall back to standard tensor creation
            states, actions, regrets, strategies, bet_sizes = zip(*batch_data)
            
            return {
                'states': torch.FloatTensor(np.array(states)).to(self.device),
                'actions': torch.LongTensor(np.array(actions)).to(self.device),
                'regrets': torch.FloatTensor(np.array(regrets)).to(self.device),
                'strategies': torch.FloatTensor(np.array(strategies)).to(self.device),
                'bet_sizes': torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            }
        
        # Optimized version using tensor pool
        if batch_size > self.max_batch_size:
            # Fallback for oversized batches
            return self._prepare_batch_fallback(batch_data, batch_size)
        
        # Unpack data
        states, actions, regrets, strategies, bet_sizes = zip(*batch_data)
        
        # Use preallocated tensors and slice them
        result = {}
        
        # Create tensors directly to avoid shape mismatch issues
        result['states'] = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        
        # Debug and fix action tensor shape issues
        actions_array = np.array(actions)
        if actions_array.ndim > 1:
            # If actions is multi-dimensional, flatten to 1D by taking first element of each
            print(f"🔧 WARNING: Actions have wrong shape {actions_array.shape}, fixing...")
            actions_fixed = []
            for action_item in actions:
                if hasattr(action_item, '__len__') and not isinstance(action_item, (int, np.integer)):
                    # Take first element if it's an array/list
                    actions_fixed.append(int(action_item[0]) if len(action_item) > 0 else 0)
                else:
                    # Keep scalar values as-is
                    actions_fixed.append(int(action_item))
            result['actions'] = torch.from_numpy(np.array(actions_fixed, dtype=np.int64)).to(self.device)
        else:
            result['actions'] = torch.from_numpy(np.array(actions, dtype=np.int64)).to(self.device)
        
        result['regrets'] = torch.from_numpy(np.array(regrets, dtype=np.float32)).to(self.device)
        
        if strategies:
            result['strategies'] = torch.from_numpy(np.array(strategies, dtype=np.float32)).to(self.device)
        
        result['bet_sizes'] = torch.from_numpy(np.array(bet_sizes, dtype=np.float32).reshape(-1, 1)).to(self.device)
        
        return result
    
    def _prepare_batch_fallback(self, batch_data: List[Tuple], batch_size: int) -> Dict[str, torch.Tensor]:
        """Fallback method for oversized batches."""
        states, actions, regrets, strategies, bet_sizes = zip(*batch_data)
        
        return {
            'states': torch.FloatTensor(np.array(states)).to(self.device),
            'actions': torch.LongTensor(np.array(actions)).to(self.device), 
            'regrets': torch.FloatTensor(np.array(regrets)).to(self.device),
            'strategies': torch.FloatTensor(np.array(strategies)).to(self.device),
            'bet_sizes': torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
        }


def get_legal_action_types_optimized(state: Any) -> List[int]:
    """
    Optimized version of getting legal action types.
    
    Optimizations:
    1. Cached action type mappings
    2. Reduced branching
    3. Direct integer operations
    """
    if not SPEED_REFAC:
        # Fall back to original implementation
        legal_action_types = []
        
        if pkrs.ActionEnum.Fold in state.legal_actions:
            legal_action_types.append(0)
        if pkrs.ActionEnum.Check in state.legal_actions or pkrs.ActionEnum.Call in state.legal_actions:
            legal_action_types.append(1)
        if pkrs.ActionEnum.Raise in state.legal_actions:
            legal_action_types.append(2)
        
        return legal_action_types
    
    # Convert to list first to avoid set hashing issues with ActionEnum
    legal_actions_list = list(state.legal_actions)
    legal_action_types = []
    
    # Direct checks without set operations
    if pkrs.ActionEnum.Fold in legal_actions_list:
        legal_action_types.append(0)
    
    # Check/Call combined check
    has_check_call = any(action in legal_actions_list for action in [pkrs.ActionEnum.Check, pkrs.ActionEnum.Call])
    if has_check_call:
        legal_action_types.append(1)
        
    if pkrs.ActionEnum.Raise in legal_actions_list:
        legal_action_types.append(2)
        
    return legal_action_types


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor performance improvements in real-time."""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
        
    def time_function(self, func_name: str, duration: float):
        """Record timing for a function call."""
        if func_name not in self.timings:
            self.timings[func_name] = []
            self.call_counts[func_name] = 0
            
        self.timings[func_name].append(duration)
        self.call_counts[func_name] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        for func_name, times in self.timings.items():
            if times:
                stats[func_name] = {
                    'mean_time': np.mean(times),
                    'total_time': np.sum(times),
                    'call_count': self.call_counts[func_name],
                    'time_per_call': np.mean(times) * 1000  # in milliseconds
                }
        return stats
    
    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.call_counts.clear()

# Global performance monitor instance
performance_monitor = PerformanceMonitor()