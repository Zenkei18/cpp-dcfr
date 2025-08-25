#ifndef DEEPCFR_MODEL_MODEL_H_
#define DEEPCFR_MODEL_MODEL_H_

#include <torch/torch.h>
#include <vector>

namespace deepcfr {

/**
 * @brief Neural network for poker decision making
 * 
 * This model implements a neural network that takes a poker state
 * as input and outputs action logits and bet sizing.
 */
class PokerNetwork : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Poker Network
     * 
     * @param input_size Size of the input features
     * @param hidden_size Size of hidden layers
     * @param num_actions Number of possible actions
     */
    PokerNetwork(int input_size = 500, int hidden_size = 256, int num_actions = 4);
    
    /**
     * @brief Forward pass through the network
     * 
     * @param x The state representation tensor
     * @return Tuple of (action_logits, bet_size)
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    
private:
    torch::nn::Sequential base_{nullptr};
    torch::nn::Linear action_head_{nullptr};
    torch::nn::Sequential sizing_head_{nullptr};
};

/**
 * @brief Convert a poker state to a tensor representation
 * 
 * @param state The poker state to encode
 * @param player_id The ID of the player for whom we're encoding
 * @return torch::Tensor The encoded state tensor
 */
torch::Tensor encode_state(const void* state, int player_id = 0);

/**
 * @brief Set the global verbosity level
 * 
 * @param verbose Whether to enable verbose output
 */
void set_verbose(bool verbose);

} // namespace deepcfr

#endif // DEEPCFR_MODEL_MODEL_H_
