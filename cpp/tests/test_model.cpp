#include <catch2/catch_all.hpp>
#include <torch/torch.h>

#include "deepcfr/model/model.h"

TEST_CASE("PokerNetwork forward pass", "[model]") {
    // Create a network with default parameters
    deepcfr::PokerNetwork network(500, 256, 4);
    
    // Create a dummy input tensor
    torch::Tensor input = torch::rand({1, 500});
    
    // Forward pass
    auto [action_logits, bet_size] = network.forward(input);
    
    // Check output shapes
    REQUIRE(action_logits.sizes() == std::vector<int64_t>{1, 4});
    REQUIRE(bet_size.sizes() == std::vector<int64_t>{1, 1});
    
    // Check bet size range (0.1 to 3.0)
    float min_bet = bet_size.min().item<float>();
    float max_bet = bet_size.max().item<float>();
    
    REQUIRE(min_bet >= 0.1f);
    REQUIRE(max_bet <= 3.0f);
}

TEST_CASE("encode_state placeholder test", "[model]") {
    // Test the placeholder implementation of encode_state
    auto encoded = deepcfr::encode_state(nullptr, 0);
    
    // Check the expected shape
    REQUIRE(encoded.sizes() == std::vector<int64_t>{500});
    
    // In the placeholder implementation, all values should be zero
    REQUIRE(encoded.sum().item<float>() == 0.0f);
}
