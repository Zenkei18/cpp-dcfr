#include <iostream>
#include <string>
#include <memory>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "deepcfr/core/deep_cfr.h"
#include "deepcfr/model/model.h"

int main(int argc, char** argv) {
    // Set up command-line argument parsing
    CLI::App app{"Play against DeepCFR Poker AI"};
    
    std::string model_path;
    int num_games = 1;
    bool verbose = false;
    bool human = false;
    
    app.add_option("--model", model_path, "Path to the trained model")->required();
    app.add_option("--num-games", num_games, "Number of games to play");
    app.add_flag("--verbose", verbose, "Enable verbose output");
    app.add_flag("--human", human, "Play as a human against the AI");
    
    // Parse command-line arguments
    CLI11_PARSE(app, argc, argv);
    
    // Set up logging
    if (verbose) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }
    
    // Set verbosity in DeepCFR modules
    deepcfr::set_verbose(verbose);
    
    // Initialize the agent
    auto device = torch::cuda::is_available() ? "cuda" : "cpu";
    spdlog::info("Using device: {}", device);
    
    auto agent = std::make_unique<deepcfr::DeepCFRAgent>(0, 6, 4, device);
    
    // Load the model
    spdlog::info("Loading model from: {}", model_path);
    agent->load_model(model_path);
    
    // Play logic would go here
    spdlog::info("Playing functionality not yet implemented in C++ version");
    spdlog::info("This is a placeholder for the actual playing implementation");
    
    return 0;
}
