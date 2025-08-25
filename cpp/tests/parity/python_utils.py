"""
Python utilities for parity testing between C++ and Python implementations.
"""
import json
import numpy as np
import pokers as pkrs

def create_game_state(n_players, button, sb, bb, stake, seed):
    """
    Create a new poker game state with the given parameters.
    Returns a serialized representation of the state.
    """
    state = pkrs.State.from_seed(
        n_players=n_players,
        button=button,
        sb=sb,
        bb=bb,
        stake=stake,
        seed=seed
    )
    return serialize_state(state)

def apply_action(state_json, action_type, amount=0.0):
    """
    Apply an action to a serialized game state.
    Returns a serialized representation of the new state.
    """
    state = deserialize_state(state_json)
    
    if action_type == 0:  # Fold
        action = pkrs.Action(pkrs.ActionEnum.Fold, 0)
    elif action_type == 1:  # Check
        action = pkrs.Action(pkrs.ActionEnum.Check, 0)
    elif action_type == 2:  # Call
        action = pkrs.Action(pkrs.ActionEnum.Call, 0)
    elif action_type == 3:  # Raise
        action = pkrs.Action(pkrs.ActionEnum.Raise, amount)
    else:
        raise ValueError(f"Unknown action type: {action_type}")
    
    new_state = state.apply_action(action)
    return serialize_state(new_state)

def get_legal_actions(state_json):
    """
    Get the legal actions for a serialized game state.
    Returns a list of action types.
    """
    state = deserialize_state(state_json)
    return [int(action) for action in state.legal_actions]

def encode_state(state_json, player_id):
    """
    Encode a state for neural network input, from the perspective of the given player.
    Returns a numpy array.
    """
    from src.core.model import encode_state as py_encode_state
    state = deserialize_state(state_json)
    encoded = py_encode_state(state, player_id)
    return encoded.tolist()

# Serialization helpers

def serialize_state(state):
    """Serialize a poker state to JSON"""
    data = {
        "current_player": state.current_player,
        "pot": state.pot,
        "min_bet": state.min_bet,
        "button": state.button,
        "stage": int(state.stage),
        "status": int(state.status),
        "final_state": state.final_state,
        "players_state": [
            {
                "player": p.player,
                "active": p.active,
                "stake": p.stake,
                "bet_chips": p.bet_chips,
                "pot_chips": p.pot_chips,
                "reward": p.reward,
                "hand": [
                    {"suit": int(card.suit), "rank": int(card.rank)}
                    for card in p.hand
                ]
            }
            for p in state.players_state
        ],
        "public_cards": [
            {"suit": int(card.suit), "rank": int(card.rank)}
            for card in state.public_cards
        ],
        "legal_actions": [int(action) for action in state.legal_actions]
    }
    
    return json.dumps(data)

def deserialize_state(state_json):
    """Deserialize a JSON string to a poker state"""
    data = json.loads(state_json)
    
    # Create a base state
    state = pkrs.State.from_seed(
        n_players=len(data["players_state"]),
        button=data["button"],
        sb=1,  # Default values, will be overridden
        bb=2,
        stake=200,
        seed=42
    )
    
    # Manually set all the state properties
    state.current_player = data["current_player"]
    state.pot = data["pot"]
    state.min_bet = data["min_bet"]
    state.button = data["button"]
    state.stage = pkrs.Stage(data["stage"])
    state.status = pkrs.StateStatus(data["status"])
    state.final_state = data["final_state"]
    
    # Set player states
    for i, p_data in enumerate(data["players_state"]):
        player = state.players_state[i]
        player.player = p_data["player"]
        player.active = p_data["active"]
        player.stake = p_data["stake"]
        player.bet_chips = p_data["bet_chips"]
        player.pot_chips = p_data["pot_chips"]
        player.reward = p_data["reward"]
        player.hand = [
            pkrs.Card(
                suit=pkrs.Suit(card["suit"]),
                rank=pkrs.Rank(card["rank"])
            )
            for card in p_data["hand"]
        ]
    
    # Set public cards
    state.public_cards = [
        pkrs.Card(
            suit=pkrs.Suit(card["suit"]),
            rank=pkrs.Rank(card["rank"])
        )
        for card in data["public_cards"]
    ]
    
    # Set legal actions
    state.legal_actions = [pkrs.ActionEnum(action) for action in data["legal_actions"]]
    
    return state
