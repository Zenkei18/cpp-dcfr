"""DeepCFR Poker AI C++ Implementation with Python Bindings."""

import os
import sys
from typing import Union, List, Dict, Any, Tuple, Optional
import numpy as np

try:
    from .deepcfr_cpp import (
        # Core classes
        State, 
        Card,
        Action,
        PlayerState,
        PrioritizedMemory,
        AgentInterface,
        RandomAgent,
        DeepCFRAgent,
        
        # Enums
        ActionEnum,
        Stage,
        StateStatus,
        
        # Functions
        set_verbose,
        encode_state,
        
        # Conversion helpers
        py_state_to_cpp,
        cpp_action_to_py
    )
    
    # Import directly to allow seamless usage from Python side
    import pokers as pkrs
except ImportError:
    import warnings
    warnings.warn(
        "Failed to import DeepCFR C++ extension. "
        "This could be because the extension has not been built, "
        "or because it is not in the Python path."
    )
    
    # Create stub classes to avoid errors
    class State: pass
    class Card: pass
    class Action: pass
    class PlayerState: pass
    class PrioritizedMemory: pass
    class AgentInterface: pass
    class RandomAgent: pass
    class DeepCFRAgent: pass
    class ActionEnum: pass
    class Stage: pass
    class StateStatus: pass
    
    def set_verbose(*args, **kwargs): pass
    def encode_state(*args, **kwargs): pass
    def py_state_to_cpp(*args, **kwargs): pass
    def cpp_action_to_py(*args, **kwargs): pass
    
    pkrs = None

__version__ = "0.1.0"

# Create adapter classes for seamless interop with the Python pokers library

class PyPokersStateAdapter:
    """Adapter to make a Python pokers.State look like a C++ State"""
    
    def __init__(self, py_state: 'pkrs.State'):
        self.py_state = py_state
        self.cpp_state = py_state_to_cpp(py_state) if py_state else None
    
    def get_current_player(self) -> int:
        return self.py_state.current_player
    
    def get_pot(self) -> float:
        return self.py_state.pot
    
    def get_min_bet(self) -> float:
        return self.py_state.min_bet
    
    def get_button(self) -> int:
        return self.py_state.button
    
    def get_stage(self) -> Stage:
        return Stage(int(self.py_state.stage))
    
    def get_status(self) -> StateStatus:
        return StateStatus(int(self.py_state.status))
    
    def is_final(self) -> bool:
        return self.py_state.final_state
    
    def get_player_state(self, player_id: int) -> 'PyPokersPlayerStateAdapter':
        return PyPokersPlayerStateAdapter(self.py_state.players_state[player_id])
    
    def get_community_cards(self) -> List[Card]:
        return [Card(int(card.suit), int(card.rank)) for card in self.py_state.public_cards]
    
    def get_legal_actions(self) -> List[ActionEnum]:
        return [ActionEnum(int(action)) for action in self.py_state.legal_actions]
    
    def apply_action(self, action: Action) -> 'PyPokersStateAdapter':
        # Convert C++ action to Python action
        action_type = action.get_type()
        amount = action.get_amount()
        
        py_action = pkrs.Action(pkrs.ActionEnum(int(action_type)), amount)
        new_py_state = self.py_state.apply_action(py_action)
        
        return PyPokersStateAdapter(new_py_state)
    
    @staticmethod
    def from_seed(n_players: int, button: int, sb: float, bb: float, 
                stake: float, seed: int) -> 'PyPokersStateAdapter':
        py_state = pkrs.State.from_seed(
            n_players=n_players,
            button=button,
            sb=sb,
            bb=bb,
            stake=stake,
            seed=seed
        )
        return PyPokersStateAdapter(py_state)

class PyPokersPlayerStateAdapter:
    """Adapter to make a Python pokers.Player look like a C++ PlayerState"""
    
    def __init__(self, py_player: 'pkrs.Player'):
        self.py_player = py_player
    
    def get_hand_cards(self) -> List[Card]:
        return [Card(int(card.suit), int(card.rank)) for card in self.py_player.hand]
    
    def get_stake(self) -> float:
        return self.py_player.stake
    
    def get_bet_chips(self) -> float:
        return self.py_player.bet_chips
    
    def get_pot_chips(self) -> float:
        return self.py_player.pot_chips
    
    def is_active(self) -> bool:
        return self.py_player.active
    
    def get_reward(self) -> float:
        return self.py_player.reward

class PyDeepCFRAgent(DeepCFRAgent):
    """Extension of DeepCFRAgent with convenience methods for Python"""
    
    def __init__(self, player_id: int, num_players: int = 6, 
                num_actions: int = 4, device: str = "cpu"):
        super().__init__(player_id, num_players, num_actions, device)
    
    def choose_action_py(self, py_state: 'pkrs.State') -> 'pkrs.Action':
        """Choose an action given a Python pokers.State"""
        adapter = PyPokersStateAdapter(py_state)
        cpp_action = self.choose_action(adapter)
        
        # Convert back to Python action
        action_type = int(cpp_action.get_type())
        amount = cpp_action.get_amount()
        
        return pkrs.Action(pkrs.ActionEnum(action_type), amount)
    
    def cfr_traverse_py(self, py_state: 'pkrs.State', iteration: int, 
                      opponent_agents: List['PyAgentWrapper'], 
                      depth: int = 0) -> float:
        """Run CFR traversal with Python pokers.State"""
        adapter = PyPokersStateAdapter(py_state)
        return self.cfr_traverse(adapter, iteration, opponent_agents, depth)

class PyAgentWrapper(AgentInterface):
    """Wrapper for Python agents to be used in the C++ code"""
    
    def __init__(self, py_agent):
        """Initialize with a Python agent that has choose_action method"""
        super().__init__()
        self.py_agent = py_agent
        self.player_id = py_agent.player_id if hasattr(py_agent, 'player_id') else -1
    
    def choose_action(self, state: State) -> Action:
        """Convert C++ state to Python, call Python agent, convert action back to C++"""
        # This would require conversion between C++ and Python types
        # For now, this is a placeholder implementation
        py_action = self.py_agent.choose_action(state.py_state)
        
        # Convert Python action to C++ action
        action_type = int(py_action.action)
        amount = py_action.amount
        
        if action_type == 0:  # Fold
            return Action.fold()
        elif action_type == 1:  # Check
            return Action.check()
        elif action_type == 2:  # Call
            return Action.call()
        elif action_type == 3:  # Raise
            return Action.raise_(amount)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def get_player_id(self) -> int:
        """Return the player ID"""
        return self.player_id