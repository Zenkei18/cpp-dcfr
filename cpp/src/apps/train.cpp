#include <iostream>
#include <string>
#include <memory>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include "deepcfr/core/deep_cfr.h"
#include "deepcfr/model/model.h"

int main(int argc, char** argv) {
    // Set up command-line argument parsing
    CLI::App app{"DeepCFR Poker AI Training"};
    
    bool verbose = false;
    int iterations = 1000;
    int traversals = 200;
    std::string save_dir = "models";
    std::string log_dir = "logs/deepcfr";
    std::string checkpoint;
    bool self_play = false;
    bool mixed = false;
    std::string checkpoint_dir = "models";
    std::string model_prefix = "t_";
    int refresh_interval = 1000;
    int num_opponents = 5;
    bool strict = false;
    
    // Add command-line options
    app.add_flag("--verbose", verbose, "Enable verbose output");
    app.add_option("--iterations", iterations, "Number of CFR iterations");
    app.add_option("--traversals", traversals, "Traversals per iteration");
    app.add_option("--save-dir", save_dir, "Directory to save models");
    app.add_option("--log-dir", log_dir, "Directory for logs");
    app.add_option("--checkpoint", checkpoint, "Path to checkpoint to continue training from");
    app.add_flag("--self-play", self_play, "Train against checkpoint instead of random agents");
    app.add_flag("--mixed", mixed, "Train against mixed checkpoints");
    app.add_option("--checkpoint-dir", checkpoint_dir, "Directory containing checkpoint models");
    app.add_option("--model-prefix", model_prefix, "Prefix for models to include in selection pool");
    app.add_option("--refresh-interval", refresh_interval, "Interval to refresh opponent pool");
    app.add_option("--num-opponents", num_opponents, "Number of checkpoint opponents to select");
    app.add_flag("--strict", strict, "Enable strict error checking that raises exceptions for invalid game states");
    
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
    
    // Print training configuration
    spdlog::info("Starting Deep CFR training with the following configuration:");
    spdlog::info("  Iterations: {}", iterations);
    spdlog::info("  Traversals per iteration: {}", traversals);
    spdlog::info("  Save directory: {}", save_dir);
    spdlog::info("  Log directory: {}", log_dir);
    
    if (!checkpoint.empty()) {
        spdlog::info("  Loading checkpoint from: {}", checkpoint);
        agent->load_model(checkpoint);
    }
    
    // Training logic would go here
    spdlog::info("Training not yet implemented in C++ version");
    spdlog::info("This is a placeholder for the actual training implementation");
    
    return 0;
}
