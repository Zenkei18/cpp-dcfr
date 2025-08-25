#include <benchmark/benchmark.h>
#include <Python.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <random>
#include <spdlog/spdlog.h>

#include "deepcfr/core/deep_cfr.h"
#include "deepcfr/model/model.h"
#include "deepcfr/pokers/pokers.h"

using json = nlohmann::json;
using namespace deepcfr;

// Helper functions for Python interop (duplicated from parity tests)
json call_python_function(const char* function_name, PyObject* args) {
    PyObject* pModule = PyImport_ImportModule("python_utils");
    if (!pModule) {
        throw std::runtime_error("Failed to import python_utils module");
    }
    
    PyObject* pFunc = PyObject_GetAttrString(pModule, function_name);
    if (!pFunc || !PyCallable_Check(pFunc)) {
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
        throw std::runtime_error(std::string("Failed to find function: ") + function_name);
    }
    
    PyObject* pResult = PyObject_CallObject(pFunc, args);
    if (!pResult) {
        Py_XDECREF(pResult);
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
        PyErr_Print();
        throw std::runtime_error(std::string("Call to ") + function_name + " failed");
    }
    
    // Convert Python result to JSON
    PyObject* pJson = PyImport_ImportModule("json");
    PyObject* pDumps = PyObject_GetAttrString(pJson, "dumps");
    PyObject* pJsonStr = PyObject_CallFunctionObjArgs(pDumps, pResult, NULL);
    
    const char* json_str = PyUnicode_AsUTF8(pJsonStr);
    json result = json::parse(json_str);
    
    Py_XDECREF(pJsonStr);
    Py_XDECREF(pDumps);
    Py_XDECREF(pJson);
    Py_XDECREF(pResult);
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
    
    return result;
}

// Initialize Python interpreter
class PythonInterpreterInit {
public:
    PythonInterpreterInit() {
        Py_Initialize();
        
        // Add current directory to Python path
        PyRun_SimpleString("import sys; sys.path.append('.')");
        
        // Import our utility module
        PyRun_SimpleString("import python_utils");
    }
    
    ~PythonInterpreterInit() {
        Py_Finalize();
    }
};

static PythonInterpreterInit python_interpreter;

// Benchmark state creation
static void BM_StateCreation_CPP(benchmark::State& state) {
    // Parameters for state creation
    int n_players = 6;
    int button = 0;
    float sb = 1.0f;
    float bb = 2.0f;
    float stake = 200.0f;
    int seed = 42;
    
    for (auto _ : state) {
        auto cpp_state = State::from_seed(n_players, button, sb, bb, stake, seed);
        benchmark::DoNotOptimize(cpp_state);
    }
}
BENCHMARK(BM_StateCreation_CPP);

static void BM_StateCreation_Python(benchmark::State& state) {
    // Parameters for state creation
    int n_players = 6;
    int button = 0;
    float sb = 1.0f;
    float bb = 2.0f;
    float stake = 200.0f;
    int seed = 42;
    
    PyObject* pArgs = PyTuple_New(6);
    PyTuple_SetItem(pArgs, 0, PyLong_FromLong(n_players));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(button));
    PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(sb));
    PyTuple_SetItem(pArgs, 3, PyFloat_FromDouble(bb));
    PyTuple_SetItem(pArgs, 4, PyFloat_FromDouble(stake));
    PyTuple_SetItem(pArgs, 5, PyLong_FromLong(seed));
    
    for (auto _ : state) {
        json py_state_json = call_python_function("create_game_state", pArgs);
        benchmark::DoNotOptimize(py_state_json);
    }
    
    Py_DECREF(pArgs);
}
BENCHMARK(BM_StateCreation_Python);

// Benchmark state encoding
static void BM_StateEncoding_CPP(benchmark::State& state) {
    // Create a test state
    auto cpp_state = State::from_seed(6, 0, 1.0f, 2.0f, 200.0f, 42);
    int player_id = 0;
    
    for (auto _ : state) {
        torch::Tensor encoded = encode_state(&cpp_state, player_id);
        benchmark::DoNotOptimize(encoded);
    }
}
BENCHMARK(BM_StateEncoding_CPP);

static void BM_StateEncoding_Python(benchmark::State& state) {
    // Create a test state
    PyObject* pArgs = PyTuple_New(6);
    PyTuple_SetItem(pArgs, 0, PyLong_FromLong(6));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(0));
    PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(1.0f));
    PyTuple_SetItem(pArgs, 3, PyFloat_FromDouble(2.0f));
    PyTuple_SetItem(pArgs, 4, PyFloat_FromDouble(200.0f));
    PyTuple_SetItem(pArgs, 5, PyLong_FromLong(42));
    
    json py_state_json = call_python_function("create_game_state", pArgs);
    Py_DECREF(pArgs);
    
    pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(py_state_json.dump().c_str()));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(0));
    
    for (auto _ : state) {
        json encoded = call_python_function("encode_state", pArgs);
        benchmark::DoNotOptimize(encoded);
    }
    
    Py_DECREF(pArgs);
}
BENCHMARK(BM_StateEncoding_Python);

// Benchmark action application
static void BM_ActionApplication_CPP(benchmark::State& state) {
    // Create a test state
    auto cpp_state = State::from_seed(6, 0, 1.0f, 2.0f, 200.0f, 42);
    auto action = Action::call();
    
    for (auto _ : state) {
        auto new_state = cpp_state.apply_action(action);
        benchmark::DoNotOptimize(new_state);
    }
}
BENCHMARK(BM_ActionApplication_CPP);

static void BM_ActionApplication_Python(benchmark::State& state) {
    // Create a test state
    PyObject* pArgs = PyTuple_New(6);
    PyTuple_SetItem(pArgs, 0, PyLong_FromLong(6));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(0));
    PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(1.0f));
    PyTuple_SetItem(pArgs, 3, PyFloat_FromDouble(2.0f));
    PyTuple_SetItem(pArgs, 4, PyFloat_FromDouble(200.0f));
    PyTuple_SetItem(pArgs, 5, PyLong_FromLong(42));
    
    json py_state_json = call_python_function("create_game_state", pArgs);
    Py_DECREF(pArgs);
    
    // Call action
    pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(py_state_json.dump().c_str()));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(2)); // Call action = 2
    
    for (auto _ : state) {
        json new_state = call_python_function("apply_action", pArgs);
        benchmark::DoNotOptimize(new_state);
    }
    
    Py_DECREF(pArgs);
}
BENCHMARK(BM_ActionApplication_Python);

// Benchmark CFR traversal
static void BM_CFRTraversal_CPP(benchmark::State& state) {
    // Create a simplified game with fewer players for faster benchmarking
    int num_players = 3;
    State cpp_state = State::from_seed(num_players, 0, 1.0f, 2.0f, 100.0f, 42);
    
    // Create agents
    std::vector<std::shared_ptr<AgentInterface>> agents;
    for (int i = 0; i < num_players; ++i) {
        agents.push_back(std::make_shared<RandomAgent>(i));
    }
    
    // Replace one agent with a DeepCFRAgent
    int agent_id = 0;
    agents[agent_id] = std::make_shared<DeepCFRAgent>(agent_id, num_players);
    
    // Run CFR traversal
    auto deep_cfr_agent = std::dynamic_pointer_cast<DeepCFRAgent>(agents[agent_id]);
    
    for (auto _ : state) {
        float value = deep_cfr_agent->cfr_traverse(cpp_state, 1, agents);
        benchmark::DoNotOptimize(value);
    }
}
BENCHMARK(BM_CFRTraversal_CPP);

// Benchmark PrioritizedMemory
static void BM_PrioritizedMemory_Add_CPP(benchmark::State& state) {
    PrioritizedMemory memory(10000);
    torch::Tensor dummy_state = torch::rand({500});
    int action_id = 0;
    float regret = 0.1f;
    
    for (auto _ : state) {
        memory.add(dummy_state, action_id, regret);
        
        // Reset if we reach capacity
        if (memory.size() >= 10000) {
            benchmark::DoNotOptimize(memory.size());
            // Create a new memory with the same capacity
            memory = PrioritizedMemory(10000);
        }
    }
}
BENCHMARK(BM_PrioritizedMemory_Add_CPP);

static void BM_PrioritizedMemory_Sample_CPP(benchmark::State& state) {
    PrioritizedMemory memory(10000);
    
    // Fill memory with data
    for (int i = 0; i < 1000; ++i) {
        torch::Tensor dummy_state = torch::rand({500});
        memory.add(dummy_state, i % 4, i * 0.01f);
    }
    
    for (auto _ : state) {
        auto [states, action_ids, regrets, indices, weights] = memory.sample(128);
        benchmark::DoNotOptimize(states);
        benchmark::DoNotOptimize(action_ids);
        benchmark::DoNotOptimize(regrets);
        benchmark::DoNotOptimize(indices);
        benchmark::DoNotOptimize(weights);
    }
}
BENCHMARK(BM_PrioritizedMemory_Sample_CPP);

// Benchmark neural network forward pass
static void BM_NeuralNetwork_Forward_CPP(benchmark::State& state) {
    PokerNetwork network(500, 256, 4);
    
    // Create batch of different sizes
    int batch_size = state.range(0);
    torch::Tensor input = torch::rand({batch_size, 500});
    
    for (auto _ : state) {
        auto [action_logits, bet_size] = network.forward(input);
        benchmark::DoNotOptimize(action_logits);
        benchmark::DoNotOptimize(bet_size);
    }
    
    // Report items/sec
    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_NeuralNetwork_Forward_CPP)->Arg(1)->Arg(32)->Arg(128)->Arg(512);

// Main function is in bench_main.cpp
