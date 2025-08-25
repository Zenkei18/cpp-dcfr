#include <benchmark/benchmark.h>
#include <torch/torch.h>

#include "deepcfr/model/model.h"

// Benchmark for neural network forward pass
static void BM_PokerNetworkForward(benchmark::State& state) {
    // Create a network with default parameters
    deepcfr::PokerNetwork network(500, 256, 4);
    
    // Create a dummy batch of input tensors
    int batch_size = state.range(0);
    torch::Tensor input = torch::rand({batch_size, 500});
    
    for (auto _ : state) {
        auto [action_logits, bet_size] = network.forward(input);
        benchmark::DoNotOptimize(action_logits);
        benchmark::DoNotOptimize(bet_size);
    }
    
    // Report throughput
    state.SetItemsProcessed(state.iterations() * batch_size);
}

// Register benchmarks with different batch sizes
BENCHMARK(BM_PokerNetworkForward)->Arg(1)->Arg(32)->Arg(128)->Arg(512);

// Benchmark for state encoding
static void BM_EncodeState(benchmark::State& state) {
    for (auto _ : state) {
        auto encoded = deepcfr::encode_state(nullptr, 0);
        benchmark::DoNotOptimize(encoded);
    }
}

BENCHMARK(BM_EncodeState);
