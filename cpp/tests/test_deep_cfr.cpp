#include <catch2/catch_all.hpp>
#include <torch/torch.h>

#include "deepcfr/core/deep_cfr.h"

TEST_CASE("PrioritizedMemory basic operations", "[deep_cfr]") {
    // Create memory buffer
    deepcfr::PrioritizedMemory memory(10, 0.6f);
    
    // Check initial state
    REQUIRE(memory.size() == 0);
    REQUIRE(memory.empty());
    
    // Add items
    for (int i = 0; i < 5; ++i) {
        auto state = torch::ones({500}) * i;
        memory.add(state, i, i * 0.1f);
    }
    
    // Check size after adding
    REQUIRE(memory.size() == 5);
    REQUIRE(!memory.empty());
    
    // Sample batch
    auto [states, action_ids, regrets, indices, weights] = memory.sample(3, 0.4f);
    
    // Check sample shapes
    REQUIRE(states.sizes() == std::vector<int64_t>{3, 500});
    REQUIRE(action_ids.sizes() == std::vector<int64_t>{3});
    REQUIRE(regrets.sizes() == std::vector<int64_t>{3});
    REQUIRE(indices.size() == 3);
    REQUIRE(weights.sizes() == std::vector<int64_t>{3});
    
    // Update priorities
    for (size_t i = 0; i < indices.size(); ++i) {
        memory.update_priority(indices[i], 10.0f);
    }
}

TEST_CASE("DeepCFRAgent initialization", "[deep_cfr]") {
    // Create agent
    deepcfr::DeepCFRAgent agent(0, 6, 4, "cpu");
    
    // Test choose_action placeholder
    int action = agent.choose_action(nullptr);
    
    // Action should be within the range of legal actions
    REQUIRE(action >= 0);
    REQUIRE(action < 4);
    
    // Test cfr_traverse placeholder
    std::vector<void*> opponent_agents(6, nullptr);
    float value = agent.cfr_traverse(nullptr, 1, opponent_agents);
    
    // The placeholder implementation returns 0.0f
    REQUIRE(value == 0.0f);
    
    // Test train_advantage_network placeholder
    float adv_loss = agent.train_advantage_network();
    
    // The placeholder implementation returns 0.0f when memory is empty
    REQUIRE(adv_loss == 0.0f);
    
    // Test train_strategy_network placeholder
    float strat_loss = agent.train_strategy_network();
    
    // The placeholder implementation returns 0.0f when memory is empty
    REQUIRE(strat_loss == 0.0f);
}
