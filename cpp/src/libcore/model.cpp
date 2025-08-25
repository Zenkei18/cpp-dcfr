#include "deepcfr/model/model.h"
#include "deepcfr/pokers/pokers.h"
#include <spdlog/spdlog.h>
#include <vector>

namespace deepcfr {

namespace {
bool VERBOSE = false;
}

void set_verbose(bool verbose) {
    VERBOSE = verbose;
    if (VERBOSE) {
        spdlog::info("Verbose mode enabled");
    }
}

PokerNetwork::PokerNetwork(int input_size, int hidden_size, int num_actions) {
    // Shared feature extraction layers
    base_ = torch::nn::Sequential(
        torch::nn::Linear(input_size, hidden_size),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_size, hidden_size),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_size, hidden_size),
        torch::nn::ReLU()
    );
    
    // Action type prediction (fold, check/call, raise)
    action_head_ = torch::nn::Linear(hidden_size, num_actions);
    
    // Continuous bet sizing prediction
    sizing_head_ = torch::nn::Sequential(
        torch::nn::Linear(hidden_size, hidden_size / 2),
        torch::nn::Tanh(),
        torch::nn::Linear(hidden_size / 2, 1),
        torch::nn::Sigmoid()  // Output between 0-1
    );
    
    // Register the submodules
    register_module("base", base_);
    register_module("action_head", action_head_);
    register_module("sizing_head", sizing_head_);
}

std::tuple<torch::Tensor, torch::Tensor> PokerNetwork::forward(torch::Tensor x) {
    // Process base features
    auto features = base_->forward(x);
    
    // Output action logits and bet sizing
    auto action_logits = action_head_->forward(features);
    auto bet_size_raw = sizing_head_->forward(features);
    
    // Scale bet size from 0.1 to 3.0
    auto bet_size = 0.1 + 2.9 * bet_size_raw;
    
    return {action_logits, bet_size};
}

torch::Tensor encode_state(const void* state_ptr, int player_id) {
    // Cast state_ptr to the proper type
    const State* state = static_cast<const State*>(state_ptr);
    
    if (VERBOSE) {
        spdlog::debug("encode_state called with player_id={}", player_id);
    }
    
    // Create a vector to hold all encoded features
    std::vector<torch::Tensor> encoded_features;
    
    // Get the number of players
    const int num_players = 6; // Assuming 6-player poker
    
    // Encode player's hole cards
    auto player_state = state->get_player_state(player_id);
    auto hand_cards = player_state.get_hand_cards();
    
    auto hand_enc = torch::zeros({52});
    for (const auto& card : hand_cards) {
        int card_idx = card.to_int();
        hand_enc[card_idx] = 1.0f;
    }
    encoded_features.push_back(hand_enc);
    
    // Encode community cards
    auto community_cards = state->get_community_cards();
    auto community_enc = torch::zeros({52});
    for (const auto& card : community_cards) {
        int card_idx = card.to_int();
        community_enc[card_idx] = 1.0f;
    }
    encoded_features.push_back(community_enc);
    
    // Encode game stage
    auto stage_enc = torch::zeros({5}); // Preflop, Flop, Turn, River, Showdown
    stage_enc[static_cast<int>(state->get_stage())] = 1.0f;
    encoded_features.push_back(stage_enc);
    
    // Get initial stake - prevent division by zero
    float initial_stake = state->get_player_state(0).get_stake();
    if (initial_stake <= 0) {
        if (VERBOSE) {
            spdlog::warn("Initial stake is zero or negative: {}", initial_stake);
        }
        initial_stake = 1.0f; // Fallback to prevent division by zero
    }
    
    // Encode pot size (normalized by initial stake)
    float pot = state->get_pot();
    auto pot_enc = torch::tensor({pot / initial_stake});
    encoded_features.push_back(pot_enc);
    
    // Encode button position
    auto button_enc = torch::zeros({num_players});
    button_enc[state->get_button()] = 1.0f;
    encoded_features.push_back(button_enc);
    
    // Encode current player
    auto current_player_enc = torch::zeros({num_players});
    current_player_enc[state->get_current_player()] = 1.0f;
    encoded_features.push_back(current_player_enc);
    
    // Encode player states
    for (int p = 0; p < num_players; ++p) {
        auto p_state = state->get_player_state(p);
        
        // Active status
        float active = p_state.is_active() ? 1.0f : 0.0f;
        
        // Current bet (normalized)
        float bet = p_state.get_bet_chips() / initial_stake;
        
        // Pot chips (normalized)
        float pot_chips = p_state.get_pot_chips() / initial_stake;
        
        // Remaining stake (normalized)
        float stake = p_state.get_stake() / initial_stake;
        
        // Combine player state features
        auto player_features = torch::tensor({active, bet, pot_chips, stake});
        encoded_features.push_back(player_features);
    }
    
    // Encode minimum bet (normalized)
    auto min_bet_enc = torch::tensor({state->get_min_bet() / initial_stake});
    encoded_features.push_back(min_bet_enc);
    
    // Encode legal actions
    auto legal_actions = state->get_legal_actions();
    auto legal_actions_enc = torch::zeros({4}); // Fold, Check, Call, Raise
    
    for (const auto& action : legal_actions) {
        legal_actions_enc[static_cast<int>(action)] = 1.0f;
    }
    encoded_features.push_back(legal_actions_enc);
    
    // For previous action encoding, we'd need to track the previous action
    // Since we don't have that here, we'll just add zeros
    auto prev_action_enc = torch::zeros({5}); // Action type (4) + amount
    encoded_features.push_back(prev_action_enc);
    
    // Concatenate all features
    return torch::cat(encoded_features);
}

} // namespace deepcfr