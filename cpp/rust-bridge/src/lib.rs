//! FFI wrapper for the pokers crate to be used from C++

use pokers::{State, Player, Card, Rank, Suit, Action, ActionEnum, Stage, StateStatus};
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int, c_float};
use std::slice;

#[cxx::bridge(namespace = "pokers")]
mod ffi {
    // Simple enums map nicely between Rust and C++
    enum ActionEnum {
        Fold = 0,
        Check = 1,
        Call = 2,
        Raise = 3,
    }

    enum Stage {
        Preflop = 0,
        Flop = 1,
        Turn = 2,
        River = 3,
        Showdown = 4,
    }

    enum StateStatus {
        Ok = 0,
        InvalidAction = 1,
        GameOver = 2,
    }

    // Opaque types that will be implemented in Rust
    extern "Rust" {
        type StateHandle;
        type ActionHandle;
        type PlayerState;

        // State management
        fn state_create(n_players: i32, button: i32, sb: f32, bb: f32, stake: f32, seed: i32) -> Box<StateHandle>;
        fn state_destroy(state: Box<StateHandle>);
        fn state_clone(state: &StateHandle) -> Box<StateHandle>;
        
        // State properties
        fn state_get_current_player(state: &StateHandle) -> i32;
        fn state_get_pot(state: &StateHandle) -> f32;
        fn state_get_min_bet(state: &StateHandle) -> f32;
        fn state_get_button(state: &StateHandle) -> i32;
        fn state_get_stage(state: &StateHandle) -> Stage;
        fn state_get_status(state: &StateHandle) -> StateStatus;
        fn state_is_final(state: &StateHandle) -> bool;
        
        // Player state access
        fn state_get_player_state(state: &StateHandle, player_id: i32) -> Box<PlayerState>;
        fn player_state_get_hand_cards(player_state: &PlayerState, cards: &mut [i32; 2]);
        fn player_state_get_stake(player_state: &PlayerState) -> f32;
        fn player_state_get_bet_chips(player_state: &PlayerState) -> f32;
        fn player_state_get_pot_chips(player_state: &PlayerState) -> f32;
        fn player_state_is_active(player_state: &PlayerState) -> bool;
        fn player_state_get_reward(player_state: &PlayerState) -> f32;
        
        // Community cards
        fn state_get_community_cards_count(state: &StateHandle) -> i32;
        fn state_get_community_cards(state: &StateHandle, cards: &mut [i32]);
        
        // Actions
        fn state_get_legal_actions(state: &StateHandle) -> Vec<ActionEnum>;
        fn action_create_fold() -> Box<ActionHandle>;
        fn action_create_check() -> Box<ActionHandle>;
        fn action_create_call() -> Box<ActionHandle>;
        fn action_create_raise(amount: f32) -> Box<ActionHandle>;
        fn action_get_type(action: &ActionHandle) -> ActionEnum;
        fn action_get_amount(action: &ActionHandle) -> f32;
        fn action_destroy(action: Box<ActionHandle>);
        
        // Game progression
        fn state_apply_action(state: &StateHandle, action: &ActionHandle) -> Box<StateHandle>;
    }
}

// Rust implementations of the FFI functions

pub struct StateHandle {
    state: State,
}

pub struct ActionHandle {
    action: Action,
}

pub struct PlayerState {
    player: Player,
}

fn state_create(n_players: i32, button: i32, sb: f32, bb: f32, stake: f32, seed: i32) -> Box<StateHandle> {
    let state = State::from_seed(
        n_players as usize,
        button as usize,
        sb,
        bb,
        stake,
        seed as u64,
    );
    
    Box::new(StateHandle { state })
}

fn state_destroy(_state: Box<StateHandle>) {
    // Box will be dropped automatically
}

fn state_clone(state: &StateHandle) -> Box<StateHandle> {
    Box::new(StateHandle { state: state.state.clone() })
}

fn state_get_current_player(state: &StateHandle) -> i32 {
    state.state.current_player as i32
}

fn state_get_pot(state: &StateHandle) -> f32 {
    state.state.pot
}

fn state_get_min_bet(state: &StateHandle) -> f32 {
    state.state.min_bet
}

fn state_get_button(state: &StateHandle) -> i32 {
    state.state.button as i32
}

fn state_get_stage(state: &StateHandle) -> ffi::Stage {
    match state.state.stage {
        Stage::Preflop => ffi::Stage::Preflop,
        Stage::Flop => ffi::Stage::Flop,
        Stage::Turn => ffi::Stage::Turn,
        Stage::River => ffi::Stage::River,
        Stage::Showdown => ffi::Stage::Showdown,
    }
}

fn state_get_status(state: &StateHandle) -> ffi::StateStatus {
    match state.state.status {
        StateStatus::Ok => ffi::StateStatus::Ok,
        StateStatus::InvalidAction => ffi::StateStatus::InvalidAction,
        StateStatus::GameOver => ffi::StateStatus::GameOver,
    }
}

fn state_is_final(state: &StateHandle) -> bool {
    state.state.final_state
}

fn state_get_player_state(state: &StateHandle, player_id: i32) -> Box<PlayerState> {
    Box::new(PlayerState { player: state.state.players_state[player_id as usize].clone() })
}

fn player_state_get_hand_cards(player_state: &PlayerState, cards: &mut [i32; 2]) {
    for (i, card) in player_state.player.hand.iter().enumerate() {
        if i < 2 {
            cards[i] = card_to_int(*card);
        }
    }
}

fn player_state_get_stake(player_state: &PlayerState) -> f32 {
    player_state.player.stake
}

fn player_state_get_bet_chips(player_state: &PlayerState) -> f32 {
    player_state.player.bet_chips
}

fn player_state_get_pot_chips(player_state: &PlayerState) -> f32 {
    player_state.player.pot_chips
}

fn player_state_is_active(player_state: &PlayerState) -> bool {
    player_state.player.active
}

fn player_state_get_reward(player_state: &PlayerState) -> f32 {
    player_state.player.reward
}

fn state_get_community_cards_count(state: &StateHandle) -> i32 {
    state.state.public_cards.len() as i32
}

fn state_get_community_cards(state: &StateHandle, cards: &mut [i32]) {
    for (i, card) in state.state.public_cards.iter().enumerate() {
        if i < cards.len() {
            cards[i] = card_to_int(*card);
        }
    }
}

fn state_get_legal_actions(state: &StateHandle) -> Vec<ffi::ActionEnum> {
    state.state.legal_actions.iter().map(|action| {
        match action {
            ActionEnum::Fold => ffi::ActionEnum::Fold,
            ActionEnum::Check => ffi::ActionEnum::Check,
            ActionEnum::Call => ffi::ActionEnum::Call,
            ActionEnum::Raise => ffi::ActionEnum::Raise,
        }
    }).collect()
}

fn action_create_fold() -> Box<ActionHandle> {
    Box::new(ActionHandle { action: Action { action: ActionEnum::Fold, amount: 0.0 } })
}

fn action_create_check() -> Box<ActionHandle> {
    Box::new(ActionHandle { action: Action { action: ActionEnum::Check, amount: 0.0 } })
}

fn action_create_call() -> Box<ActionHandle> {
    Box::new(ActionHandle { action: Action { action: ActionEnum::Call, amount: 0.0 } })
}

fn action_create_raise(amount: f32) -> Box<ActionHandle> {
    Box::new(ActionHandle { action: Action { action: ActionEnum::Raise, amount } })
}

fn action_get_type(action: &ActionHandle) -> ffi::ActionEnum {
    match action.action.action {
        ActionEnum::Fold => ffi::ActionEnum::Fold,
        ActionEnum::Check => ffi::ActionEnum::Check,
        ActionEnum::Call => ffi::ActionEnum::Call,
        ActionEnum::Raise => ffi::ActionEnum::Raise,
    }
}

fn action_get_amount(action: &ActionHandle) -> f32 {
    action.action.amount
}

fn action_destroy(_action: Box<ActionHandle>) {
    // Box will be dropped automatically
}

fn state_apply_action(state: &StateHandle, action: &ActionHandle) -> Box<StateHandle> {
    let new_state = state.state.apply_action(action.action);
    Box::new(StateHandle { state: new_state })
}

// Helper functions

fn card_to_int(card: Card) -> i32 {
    (card.suit as i32) * 13 + (card.rank as i32)
}

fn int_to_card(value: i32) -> Card {
    let suit = (value / 13) as u8;
    let rank = (value % 13) as u8;
    Card { suit: Suit::from(suit), rank: Rank::from(rank) }
}
