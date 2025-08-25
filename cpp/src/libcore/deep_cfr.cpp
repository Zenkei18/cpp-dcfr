#include "deepcfr/core/deep_cfr.h"
#include "deepcfr/model/model.h"
#include "deepcfr/pokers/pokers.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace deepcfr {

// PrioritizedMemory Implementation

PrioritizedMemory::PrioritizedMemory(size_t capacity, float alpha)
    : capacity_(capacity), 
      alpha_(alpha), 
      max_priority_(1.0f), 
      position_(0) {
    buffer_.reserve(capacity_);
    priorities_.reserve(capacity_);
}

void PrioritizedMemory::add(const torch::Tensor& state, int action_id, float regret, float priority) {
    if (priority < 0.0f) {
        priority = max_priority_;
    }
    
    Experience exp{state, action_id, regret};
    
    if (buffer_.size() < capacity_) {
        buffer_.push_back(exp);
        priorities_.push_back(std::pow(priority, alpha_));
    } else {
        buffer_[position_] = exp;
        priorities_[position_] = std::pow(priority, alpha_);
    }
    
    position_ = (position_ + 1) % capacity_;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<size_t>, torch::Tensor> 
PrioritizedMemory::sample(size_t batch_size, float beta) {
    if (buffer_.empty()) {
        throw std::runtime_error("Cannot sample from an empty buffer");
    }
    
    if (buffer_.size() < batch_size) {
        batch_size = buffer_.size();
    }
    
    // Calculate sum of priorities
    float total_priority = 0.0f;
    for (float priority : priorities_) {
        total_priority += priority;
    }
    
    // Calculate sampling probabilities
    std::vector<float> probabilities;
    probabilities.reserve(buffer_.size());
    
    for (float priority : priorities_) {
        probabilities.push_back(priority / total_priority);
    }
    
    // Sample indices based on priorities
    std::vector<size_t> indices;
    indices.reserve(batch_size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<size_t> dist(probabilities.begin(), probabilities.end());
    
    for (size_t i = 0; i < batch_size; ++i) {
        indices.push_back(dist(gen));
    }
    
    // Create output tensors
    std::vector<torch::Tensor> states;
    std::vector<int64_t> action_ids;
    std::vector<float> regrets;
    std::vector<float> weights;
    
    states.reserve(batch_size);
    action_ids.reserve(batch_size);
    regrets.reserve(batch_size);
    weights.reserve(batch_size);
    
    // Calculate importance sampling weights
    for (size_t idx : indices) {
        // P(i) = p_i^α / sum_k p_k^α
        // weight = (1/N * 1/P(i))^β = (N*P(i))^-β
        float sample_prob = probabilities[idx];
        float weight = std::pow(buffer_.size() * sample_prob, -beta);
        
        states.push_back(buffer_[idx].state);
        action_ids.push_back(buffer_[idx].action_id);
        regrets.push_back(buffer_[idx].regret);
        weights.push_back(weight);
    }
    
    // Normalize weights to have maximum weight = 1
    float max_weight = *std::max_element(weights.begin(), weights.end());
    for (auto& weight : weights) {
        weight /= max_weight;
    }
    
    // Stack tensors
    auto states_tensor = torch::stack(states);
    auto action_ids_tensor = torch::tensor(action_ids);
    auto regrets_tensor = torch::tensor(regrets);
    auto weights_tensor = torch::tensor(weights);
    
    return {states_tensor, action_ids_tensor, regrets_tensor, indices, weights_tensor};
}

void PrioritizedMemory::update_priority(size_t index, float priority) {
    if (index >= buffer_.size()) {
        throw std::out_of_range("Index out of range");
    }
    
    // Update the priority
    priorities_[index] = std::pow(std::max(priority, 1e-6f), alpha_);
    
    // Update max priority if needed
    max_priority_ = std::max(max_priority_, priority);
}

size_t PrioritizedMemory::size() const {
    return buffer_.size();
}

bool PrioritizedMemory::empty() const {
    return buffer_.empty();
}

// Random Agent Implementation

RandomAgent::RandomAgent(int player_id)
    : player_id_(player_id),
      rng_(std::random_device{}()) {
}

Action RandomAgent::choose_action(const State& state) {
    // Get legal actions
    auto legal_actions = state.get_legal_actions();
    
    if (legal_actions.empty()) {
        throw std::runtime_error("No legal actions available");
    }
    
    // Choose a random action
    std::uniform_int_distribution<size_t> dist(0, legal_actions.size() - 1);
    auto action_type = legal_actions[dist(rng_)];
    
    // Create the action
    switch (action_type) {
        case ActionEnum::Fold:
            return Action::fold();
        case ActionEnum::Check:
            return Action::check();
        case ActionEnum::Call:
            return Action::call();
        case ActionEnum::Raise: {
            // For raise, we need to choose a random amount
            float min_bet = state.get_min_bet();
            float max_bet = state.get_player_state(player_id_).get_stake();
            
            // If max_bet is less than min_bet, raise to max_bet (all-in)
            if (max_bet <= min_bet) {
                return Action::raise(max_bet);
            }
            
            // Otherwise, choose a random amount between min_bet and max_bet
            std::uniform_real_distribution<float> amount_dist(min_bet, max_bet);
            return Action::raise(amount_dist(rng_));
        }
        default:
            throw std::runtime_error("Unknown action type");
    }
}

// DeepCFRAgent Implementation

DeepCFRAgent::DeepCFRAgent(int player_id, int num_players, int num_actions, const std::string& device)
    : player_id_(player_id),
      num_players_(num_players),
      num_actions_(num_actions),
      device_(device),
      iteration_count_(0),
      advantage_memory_(10000000),  // 10M capacity
      rng_(std::random_device{}()) {
    
    // Create neural networks
    advantage_net_ = std::make_shared<PokerNetwork>(500, 256, num_actions);
    strategy_net_ = std::make_shared<PokerNetwork>(500, 256, num_actions);
    
    // Move to the specified device
    advantage_net_->to(torch::Device(device));
    strategy_net_->to(torch::Device(device));
    
    // Create optimizers
    advantage_optimizer_ = std::make_unique<torch::optim::Adam>(
        advantage_net_->parameters(), torch::optim::AdamOptions(0.001));
    
    strategy_optimizer_ = std::make_unique<torch::optim::Adam>(
        strategy_net_->parameters(), torch::optim::AdamOptions(0.001));
}

Action DeepCFRAgent::choose_action(const State& state) {
    // Check if it's our turn
    if (state.get_current_player() != player_id_) {
        throw std::runtime_error("Not our turn to act");
    }
    
    // Get legal actions
    std::vector<int> legal_action_ids = get_legal_action_ids(state);
    
    if (legal_action_ids.empty()) {
        throw std::runtime_error("No legal actions available");
    }
    
    // Encode state
    torch::Tensor state_tensor = torch::tensor(encode_state(&state, player_id_)).to(device_);
    
    // Forward pass through strategy network
    torch::NoGradGuard no_grad;
    auto [action_logits, _] = std::dynamic_pointer_cast<PokerNetwork>(strategy_net_)->forward(state_tensor.unsqueeze(0));
    
    // Get probabilities for legal actions
    torch::Tensor probs = torch::softmax(action_logits, 1)[0];
    
    // Mask out illegal actions
    torch::Tensor masked_probs = torch::zeros_like(probs);
    for (int action_id : legal_action_ids) {
        masked_probs[action_id] = probs[action_id];
    }
    
    // Renormalize
    if (masked_probs.sum().item<float>() > 0) {
        masked_probs = masked_probs / masked_probs.sum();
    } else {
        // If all probabilities are zero, use uniform distribution
        for (int action_id : legal_action_ids) {
            masked_probs[action_id] = 1.0f / legal_action_ids.size();
        }
    }
    
    // Convert to std::vector for sampling
    std::vector<float> probs_vec(masked_probs.data_ptr<float>(), 
                               masked_probs.data_ptr<float>() + masked_probs.numel());
    
    // Sample from the distribution
    std::discrete_distribution<int> dist(probs_vec.begin(), probs_vec.end());
    int action_id = dist(rng_);
    
    // Convert to poker action
    return action_id_to_pokers_action(action_id, state);
}

float DeepCFRAgent::cfr_traverse(const State& state, int iteration, 
                               const std::vector<std::shared_ptr<AgentInterface>>& opponent_agents, 
                               int depth) {
    // Add recursion depth protection
    constexpr int max_depth = 1000;
    if (depth > max_depth) {
        spdlog::warn("Max recursion depth reached ({}). Returning zero value.", max_depth);
        return 0.0f;
    }
    
    // Check if we're at a terminal state
    if (state.is_final()) {
        // Return payoff for the trained agent
        return state.get_player_state(player_id_).get_reward();
    }
    
    int current_player = state.get_current_player();
    
    // If it's the trained agent's turn
    if (current_player == player_id_) {
        // Get legal actions
        std::vector<int> legal_action_ids = get_legal_action_ids(state);
        
        if (legal_action_ids.empty()) {
            spdlog::warn("No legal actions found for player {} at depth {}", current_player, depth);
            return 0.0f;
        }
        
        // Encode state from this agent's perspective
        torch::Tensor state_tensor = torch::tensor(encode_state(&state, player_id_)).to(device_);
        
        // Get advantages from network
        torch::Tensor advantages;
        {
            torch::NoGradGuard no_grad;
            auto [adv, _] = std::dynamic_pointer_cast<PokerNetwork>(advantage_net_)->forward(state_tensor.unsqueeze(0));
            advantages = adv[0];
        }
        
        // Use regret matching to compute strategy
        std::vector<float> advantages_vec(advantages.data_ptr<float>(), 
                                        advantages.data_ptr<float>() + advantages.numel());
        std::vector<float> advantages_masked(num_actions_, 0.0f);
        
        for (int a : legal_action_ids) {
            advantages_masked[a] = std::max(advantages_vec[a], 0.0f);
        }
        
        // Choose an action based on the strategy
        std::vector<float> strategy(num_actions_, 0.0f);
        float sum_advantages = std::accumulate(advantages_masked.begin(), advantages_masked.end(), 0.0f);
        
        if (sum_advantages > 0.0f) {
            for (int a = 0; a < num_actions_; ++a) {
                strategy[a] = advantages_masked[a] / sum_advantages;
            }
        } else {
            // If all advantages are negative or zero, use uniform strategy
            for (int a : legal_action_ids) {
                strategy[a] = 1.0f / legal_action_ids.size();
            }
        }
        
        // Choose actions and traverse
        std::vector<float> action_values(num_actions_, 0.0f);
        
        for (int action_id : legal_action_ids) {
            try {
                // Convert action_id to poker action
                Action poker_action = action_id_to_pokers_action(action_id, state);
                
                // Apply the action
                State new_state = state.apply_action(poker_action);
                
                // Check if the action was valid
                if (new_state.get_status() != StateStatus::Ok) {
                    spdlog::warn("Invalid action {} at depth {}. Status: {}", 
                              action_id, depth, static_cast<int>(new_state.get_status()));
                    continue;
                }
                
                // Recurse
                action_values[action_id] = cfr_traverse(new_state, iteration, opponent_agents, depth + 1);
            } catch (const std::exception& e) {
                spdlog::error("Error in traversal for action {}: {}", action_id, e.what());
                action_values[action_id] = 0.0f;
            }
        }
        
        // Compute expected value
        float ev = 0.0f;
        for (int a : legal_action_ids) {
            ev += strategy[a] * action_values[a];
        }
        
        // Calculate normalization factor
        float max_abs_val = std::max(
            std::abs(*std::max_element(action_values.begin(), action_values.end())),
            std::abs(*std::min_element(action_values.begin(), action_values.end()))
        );
        max_abs_val = std::max(max_abs_val, 1.0f);
        
        // Calculate regrets and add to memory
        for (int action_id : legal_action_ids) {
            // Calculate regret
            float regret = action_values[action_id] - ev;
            
            // Normalize and clip regret
            float normalized_regret = regret / max_abs_val;
            float clipped_regret = std::clamp(normalized_regret, -10.0f, 10.0f);
            
            // Apply scaling
            float scale_factor = (iteration > 1) ? std::sqrt(iteration) : 1.0f;
            
            // Add to advantage memory
            advantage_memory_.add(
                state_tensor,
                action_id,
                clipped_regret * scale_factor
            );
        }
        
        // Add to strategy memory
        torch::Tensor strategy_tensor = torch::tensor(strategy);
        strategy_memory_.emplace_back(state_tensor, strategy_tensor, iteration);
        
        return ev;
    } 
    // If it's another player's turn (opponent agent)
    else {
        try {
            // Check if we have an opponent agent for this position
            if (current_player < 0 || current_player >= opponent_agents.size() || 
                !opponent_agents[current_player]) {
                spdlog::warn("No opponent agent for position {}", current_player);
                return 0.0f;
            }
            
            // Let the opponent choose an action
            Action action = opponent_agents[current_player]->choose_action(state);
            
            // Apply the action
            State new_state = state.apply_action(action);
            
            // Check if the action was valid
            if (new_state.get_status() != StateStatus::Ok) {
                spdlog::warn("Opponent agent made invalid action at depth {}. Status: {}", 
                           depth, static_cast<int>(new_state.get_status()));
                return 0.0f;
            }
            
            // Recurse
            return cfr_traverse(new_state, iteration, opponent_agents, depth + 1);
        } catch (const std::exception& e) {
            spdlog::error("Error in opponent agent traversal: {}", e.what());
            return 0.0f;
        }
    }
}

float DeepCFRAgent::train_advantage_network() {
    // Skip training if the memory is empty
    if (advantage_memory_.empty()) {
        spdlog::warn("Cannot train advantage network: memory is empty");
        return 0.0f;
    }
    
    // Get device
    auto device = torch::Device(device_);
    
    // Initialize loss accumulator
    float total_loss = 0.0f;
    int num_batches = 0;
    
    // Training loop
    for (int i = 0; i < 2000; ++i) {  // 2000 batches
        // Sample a batch
        const size_t batch_size = 512;
        auto [states, actions, regrets, indices, weights] = advantage_memory_.sample(batch_size);
        
        // Move to device
        states = states.to(device);
        actions = actions.to(device);
        regrets = regrets.to(device);
        weights = weights.to(device);
        
        // Forward pass
        advantage_optimizer_->zero_grad();
        auto [action_logits, _] = std::dynamic_pointer_cast<PokerNetwork>(advantage_net_)->forward(states);
        
        // Calculate loss
        auto pred_regrets = action_logits.gather(1, actions.unsqueeze(1)).squeeze(1);
        auto loss = torch::nn::functional::mse_loss(pred_regrets, regrets, torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone));
        
        // Apply importance sampling weights
        loss = (loss * weights).mean();
        
        // Backward pass and optimize
        loss.backward();
        advantage_optimizer_->step();
        
        // Accumulate loss
        total_loss += loss.item<float>();
        ++num_batches;
    }
    
    // Return average loss
    return total_loss / num_batches;
}

float DeepCFRAgent::train_strategy_network() {
    // Skip training if the memory is empty
    if (strategy_memory_.empty()) {
        spdlog::warn("Cannot train strategy network: memory is empty");
        return 0.0f;
    }
    
    // Get device
    auto device = torch::Device(device_);
    
    // Initialize loss accumulator
    float total_loss = 0.0f;
    int num_batches = 0;
    
    // Training loop
    for (int i = 0; i < 500; ++i) {  // 500 batches
        // Prepare batch
        const size_t batch_size = 512;
        
        // Sample batch from strategy memory
        std::vector<torch::Tensor> states;
        std::vector<torch::Tensor> strategies;
        std::vector<int> iterations;
        
        for (int j = 0; j < batch_size && !strategy_memory_.empty(); ++j) {
            // Random index
            std::uniform_int_distribution<size_t> dist(0, strategy_memory_.size() - 1);
            size_t idx = dist(rng_);
            
            // Get experience
            auto [state, strategy, iteration] = strategy_memory_[idx];
            
            states.push_back(state);
            strategies.push_back(strategy);
            iterations.push_back(iteration);
        }
        
        if (states.empty()) {
            break;
        }
        
        // Stack tensors
        auto states_tensor = torch::stack(states).to(device);
        auto strategies_tensor = torch::stack(strategies).to(device);
        
        // Forward pass
        strategy_optimizer_->zero_grad();
        auto [action_logits, _] = std::dynamic_pointer_cast<PokerNetwork>(strategy_net_)->forward(states_tensor);
        
        // Calculate loss
        auto pred_strategies = torch::softmax(action_logits, 1);
        auto loss = torch::nn::functional::mse_loss(pred_strategies, strategies_tensor);
        
        // Backward pass and optimize
        loss.backward();
        strategy_optimizer_->step();
        
        // Accumulate loss
        total_loss += loss.item<float>();
        ++num_batches;
    }
    
    // Return average loss
    return num_batches > 0 ? total_loss / num_batches : 0.0f;
}

void DeepCFRAgent::save_model(const std::string& path) {
    // Save advantage network
    torch::save(advantage_net_, path + "_advantage.pt");
    
    // Save strategy network
    torch::save(strategy_net_, path + "_strategy.pt");
    
    spdlog::info("Model saved to {}", path);
}

void DeepCFRAgent::load_model(const std::string& path) {
    // Load advantage network
    torch::load(advantage_net_, path + "_advantage.pt");
    
    // Load strategy network
    torch::load(strategy_net_, path + "_strategy.pt");
    
    spdlog::info("Model loaded from {}", path);
}

std::vector<int> DeepCFRAgent::get_legal_action_ids(const State& state) {
    // Get legal actions from the state
    std::vector<ActionEnum> legal_actions = state.get_legal_actions();
    
    // Convert to action IDs
    std::vector<int> legal_action_ids;
    legal_action_ids.reserve(legal_actions.size());
    
    for (ActionEnum action : legal_actions) {
        legal_action_ids.push_back(static_cast<int>(action));
    }
    
    return legal_action_ids;
}

Action DeepCFRAgent::action_id_to_pokers_action(int action_id, const State& state) {
    switch (static_cast<ActionEnum>(action_id)) {
        case ActionEnum::Fold:
            return Action::fold();
        case ActionEnum::Check:
            return Action::check();
        case ActionEnum::Call:
            return Action::call();
        case ActionEnum::Raise: {
            // For raise actions, we need to determine the amount
            // In the full implementation, this would use the bet sizing network
            
            // For now, use a fixed bet size strategy
            float min_bet = state.get_min_bet();
            float player_stake = state.get_player_state(player_id_).get_stake();
            
            // Simple pot-relative betting (1/2 pot, pot, 2x pot)
            float pot_size = state.get_pot();
            std::vector<float> bet_options = {
                min_bet,                 // Minimum bet
                0.5f * pot_size,         // Half pot
                pot_size,                // Pot-sized bet
                2.0f * pot_size          // 2x pot
            };
            
            // Filter options that are larger than the player's stack
            std::vector<float> valid_bets;
            for (float bet : bet_options) {
                if (bet <= player_stake) {
                    valid_bets.push_back(bet);
                }
            }
            
            // If no valid bets (or only min bet), just use min bet or all-in
            if (valid_bets.empty() || (valid_bets.size() == 1 && valid_bets[0] == min_bet)) {
                return Action::raise(std::min(min_bet, player_stake));
            }
            
            // Choose a random bet size from the valid options
            std::uniform_int_distribution<size_t> dist(0, valid_bets.size() - 1);
            float bet_amount = valid_bets[dist(rng_)];
            
            return Action::raise(bet_amount);
        }
        default:
            throw std::runtime_error("Unknown action ID");
    }
}

} // namespace deepcfr