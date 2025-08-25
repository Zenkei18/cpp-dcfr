#include <benchmark/benchmark.h>
#include <torch/torch.h>

#include "deepcfr/core/deep_cfr.h"

// Benchmark for PrioritizedMemory add
static void BM_PrioritizedMemoryAdd(benchmark::State& state) {
    // Create memory buffer
    deepcfr::PrioritizedMemory memory(state.range(0));
    
    // Create dummy state tensor
    torch::Tensor dummy_state = torch::rand({500});
    
    for (auto _ : state) {
        memory.add(dummy_state, 0, 0.1f);
        // Reset if we reach capacity
        if (memory.size() >= state.range(0)) {
            benchmark::DoNotOptimize(memory.size());
            // Create a new memory with the same capacity
            memory = deepcfr::PrioritizedMemory(state.range(0));
        }
    }
}

// Register benchmarks with different capacities
BENCHMARK(BM_PrioritizedMemoryAdd)->Arg(1000)->Arg(10000)->Arg(100000);

// Benchmark for PrioritizedMemory sample
static void BM_PrioritizedMemorySample(benchmark::State& state) {
    // Create memory buffer
    deepcfr::PrioritizedMemory memory(100000);
    
    // Fill with dummy data
    for (int i = 0; i < 10000; ++i) {
        torch::Tensor dummy_state = torch::rand({500});
        memory.add(dummy_state, i % 4, i * 0.01f);
    }
    
    // Sample batch size
    int batch_size = state.range(0);
    
    for (auto _ : state) {
        auto [states, action_ids, regrets, indices, weights] = memory.sample(batch_size);
        benchmark::DoNotOptimize(states);
        benchmark::DoNotOptimize(action_ids);
        benchmark::DoNotOptimize(regrets);
        benchmark::DoNotOptimize(indices);
        benchmark::DoNotOptimize(weights);
    }
}

// Register benchmarks with different batch sizes
BENCHMARK(BM_PrioritizedMemorySample)->Arg(32)->Arg(128)->Arg(512);

// Benchmark for DeepCFRAgent choose_action
static void BM_DeepCFRAgentChooseAction(benchmark::State& state) {
    // Create agent
    deepcfr::DeepCFRAgent agent(0, 6, 4, "cpu");
    
    for (auto _ : state) {
        int action = agent.choose_action(nullptr);
        benchmark::DoNotOptimize(action);
    }
}

BENCHMARK(BM_DeepCFRAgentChooseAction);
