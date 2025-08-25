#ifndef DEEPCFR_CORE_DEEP_CFR_H_
#define DEEPCFR_CORE_DEEP_CFR_H_

#include <memory>
#include <string>
#include <vector>
#include <random>
#include <deque>

#include <torch/torch.h>
#include "deepcfr/pokers/pokers.h"

namespace deepcfr {

// Forward declarations
class State;
class Action;

/**
 * @brief Prioritized Experience Replay memory buffer for Deep CFR training
 * 
 * This class implements a memory buffer with prioritized experience replay
 * for more efficient training of the Deep CFR agent.
 */
class PrioritizedMemory {
public:
    /**
     * @brief Construct a new Prioritized Memory buffer
     * 
     * @param capacity Maximum number of experiences to store
     * @param alpha Priority exponent parameter (0 = no prioritization, 1 = full prioritization)
     */
    PrioritizedMemory(size_t capacity, float alpha = 0.6f);

    /**
     * @brief Add a new experience to the buffer
     * 
     * @param state State representation
     * @param action_id Action ID
     * @param regret Regret value
     * @param priority Optional explicit priority value
     */
    void add(const torch::Tensor& state, int action_id, float regret, float priority = -1.0f);
    
    /**
     * @brief Sample a batch of experiences based on their priorities
     * 
     * @param batch_size Number of experiences to sample
     * @param beta Importance sampling exponent
     * @return Tuple of (states, action_ids, regrets, indices, weights)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<size_t>, torch::Tensor> 
    sample(size_t batch_size, float beta = 0.4f);
    
    /**
     * @brief Update the priority of an experience
     * 
     * @param index Index of the experience to update
     * @param priority New priority value
     */
    void update_priority(size_t index, float priority);
    
    /**
     * @brief Get the number of experiences in the buffer
     * 
     * @return size_t Number of experiences
     */
    size_t size() const;
    
    /**
     * @brief Check if the buffer is empty
     * 
     * @return true if empty, false otherwise
     */
    bool empty() const;

private:
    struct Experience {
        torch::Tensor state;
        int action_id;
        float regret;
    };

    size_t capacity_;
    float alpha_;
    float max_priority_;
    std::vector<Experience> buffer_;
    std::vector<float> priorities_;
    size_t position_;
};

// Interface for agents that can choose actions
class AgentInterface {
public:
    virtual ~AgentInterface() = default;
    virtual Action choose_action(const State& state) = 0;
    virtual int get_player_id() const = 0;
};

/**
 * @brief Deep Counterfactual Regret Minimization agent for poker
 * 
 * This class implements a Deep CFR agent that uses neural networks to approximate
 * regret values and average strategy for playing poker.
 */
class DeepCFRAgent : public AgentInterface {
public:
    /**
     * @brief Construct a new Deep CFR Agent
     * 
     * @param player_id ID of the player this agent represents
     * @param num_players Total number of players in the game
     * @param num_actions Number of possible actions
     * @param device Device to run computations on ("cpu" or "cuda")
     */
    DeepCFRAgent(int player_id, int num_players = 6, int num_actions = 4, 
                const std::string& device = "cpu");
    
    /**
     * @brief Choose an action in the given state
     * 
     * @param state Current game state
     * @return The chosen action
     */
    Action choose_action(const State& state) override;
    
    /**
     * @brief Get the player ID
     * 
     * @return int Player ID
     */
    int get_player_id() const override { return player_id_; }
    
    /**
     * @brief Run CFR traversals to collect data
     * 
     * @param state Initial game state
     * @param iteration Current iteration number
     * @param opponent_agents Opponent agents for traversal
     * @param depth Current recursion depth
     * @return Expected value from this state
     */
    float cfr_traverse(const State& state, int iteration, 
                      const std::vector<std::shared_ptr<AgentInterface>>& opponent_agents, 
                      int depth = 0);
    
    /**
     * @brief Train the advantage network using collected data
     * 
     * @return Average loss value
     */
    float train_advantage_network();
    
    /**
     * @brief Train the strategy network using collected data
     * 
     * @return Average loss value
     */
    float train_strategy_network();
    
    /**
     * @brief Save the model to a file
     * 
     * @param path Path to save the model
     */
    void save_model(const std::string& path);
    
    /**
     * @brief Load a model from a file
     * 
     * @param path Path to load the model from
     */
    void load_model(const std::string& path);
    
    /**
     * @brief Get the legal action IDs from a state
     * 
     * @param state Poker game state
     * @return Vector of legal action IDs
     */
    std::vector<int> get_legal_action_ids(const State& state);
    
    /**
     * @brief Convert an action ID to a poker action
     * 
     * @param action_id Action ID
     * @param state Current game state
     * @return Action The corresponding poker action
     */
    Action action_id_to_pokers_action(int action_id, const State& state);
    
private:
    int player_id_;
    int num_players_;
    int num_actions_;
    std::string device_;
    int iteration_count_;
    
    // Neural networks
    std::shared_ptr<torch::nn::Module> advantage_net_;
    std::shared_ptr<torch::nn::Module> strategy_net_;
    
    // Optimizers
    std::unique_ptr<torch::optim::Optimizer> advantage_optimizer_;
    std::unique_ptr<torch::optim::Optimizer> strategy_optimizer_;
    
    // Memory buffers
    PrioritizedMemory advantage_memory_;
    std::deque<std::tuple<torch::Tensor, torch::Tensor, int>> strategy_memory_;
    
    // Random number generator
    std::mt19937 rng_;
};

/**
 * @brief Random agent that selects actions uniformly at random
 */
class RandomAgent : public AgentInterface {
public:
    /**
     * @brief Construct a new Random Agent
     * 
     * @param player_id ID of the player this agent represents
     */
    explicit RandomAgent(int player_id);
    
    /**
     * @brief Choose a random legal action
     * 
     * @param state Current game state
     * @return The chosen action
     */
    Action choose_action(const State& state) override;
    
    /**
     * @brief Get the player ID
     * 
     * @return int Player ID
     */
    int get_player_id() const override { return player_id_; }
    
private:
    int player_id_;
    std::mt19937 rng_;
};

} // namespace deepcfr

#endif // DEEPCFR_CORE_DEEP_CFR_H_