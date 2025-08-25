#include <catch2/catch_all.hpp>
#include <Python.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <vector>

#include "deepcfr/pokers/pokers.h"
#include "deepcfr/model/model.h"

using json = nlohmann::json;
using namespace deepcfr;

// Helper functions for Python interop
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

// Compare C++ and Python state creation
TEST_CASE("State creation parity", "[parity]") {
    // Parameters for state creation
    int n_players = 6;
    int button = 0;
    float sb = 1.0f;
    float bb = 2.0f;
    float stake = 200.0f;
    int seed = 42;
    
    // Create state with C++
    State cpp_state = State::from_seed(n_players, button, sb, bb, stake, seed);
    
    // Create state with Python
    PyObject* pArgs = PyTuple_New(6);
    PyTuple_SetItem(pArgs, 0, PyLong_FromLong(n_players));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(button));
    PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(sb));
    PyTuple_SetItem(pArgs, 3, PyFloat_FromDouble(bb));
    PyTuple_SetItem(pArgs, 4, PyFloat_FromDouble(stake));
    PyTuple_SetItem(pArgs, 5, PyLong_FromLong(seed));
    
    json py_state_json = call_python_function("create_game_state", pArgs);
    Py_DECREF(pArgs);
    
    // Compare state properties
    REQUIRE(cpp_state.get_current_player() == py_state_json["current_player"]);
    REQUIRE(cpp_state.get_pot() == Approx(py_state_json["pot"].get<float>()));
    REQUIRE(cpp_state.get_min_bet() == Approx(py_state_json["min_bet"].get<float>()));
    REQUIRE(cpp_state.get_button() == py_state_json["button"]);
    REQUIRE(static_cast<int>(cpp_state.get_stage()) == py_state_json["stage"]);
    REQUIRE(static_cast<int>(cpp_state.get_status()) == py_state_json["status"]);
    REQUIRE(cpp_state.is_final() == py_state_json["final_state"]);
    
    // Compare player states
    for (int i = 0; i < n_players; ++i) {
        auto cpp_player = cpp_state.get_player_state(i);
        auto py_player = py_state_json["players_state"][i];
        
        REQUIRE(cpp_player.is_active() == py_player["active"]);
        REQUIRE(cpp_player.get_stake() == Approx(py_player["stake"].get<float>()));
        REQUIRE(cpp_player.get_bet_chips() == Approx(py_player["bet_chips"].get<float>()));
        REQUIRE(cpp_player.get_pot_chips() == Approx(py_player["pot_chips"].get<float>()));
        REQUIRE(cpp_player.get_reward() == Approx(py_player["reward"].get<float>()));
        
        // Compare hand cards
        auto cpp_hand = cpp_player.get_hand_cards();
        REQUIRE(cpp_hand.size() == py_player["hand"].size());
        
        for (size_t j = 0; j < cpp_hand.size(); ++j) {
            REQUIRE(cpp_hand[j].suit == py_player["hand"][j]["suit"]);
            REQUIRE(cpp_hand[j].rank == py_player["hand"][j]["rank"]);
        }
    }
    
    // Compare community cards
    auto cpp_community = cpp_state.get_community_cards();
    REQUIRE(cpp_community.size() == py_state_json["public_cards"].size());
    
    for (size_t i = 0; i < cpp_community.size(); ++i) {
        REQUIRE(cpp_community[i].suit == py_state_json["public_cards"][i]["suit"]);
        REQUIRE(cpp_community[i].rank == py_state_json["public_cards"][i]["rank"]);
    }
    
    // Compare legal actions
    auto cpp_legal = cpp_state.get_legal_actions();
    REQUIRE(cpp_legal.size() == py_state_json["legal_actions"].size());
    
    for (size_t i = 0; i < cpp_legal.size(); ++i) {
        REQUIRE(static_cast<int>(cpp_legal[i]) == py_state_json["legal_actions"][i]);
    }
}

// Test applying actions to states
TEST_CASE("Action application parity", "[parity]") {
    // Parameters for state creation
    int n_players = 6;
    int button = 0;
    float sb = 1.0f;
    float bb = 2.0f;
    float stake = 200.0f;
    int seed = 42;
    
    // Create initial state with C++
    State cpp_state = State::from_seed(n_players, button, sb, bb, stake, seed);
    
    // Create initial state with Python
    PyObject* pArgs = PyTuple_New(6);
    PyTuple_SetItem(pArgs, 0, PyLong_FromLong(n_players));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(button));
    PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(sb));
    PyTuple_SetItem(pArgs, 3, PyFloat_FromDouble(bb));
    PyTuple_SetItem(pArgs, 4, PyFloat_FromDouble(stake));
    PyTuple_SetItem(pArgs, 5, PyLong_FromLong(seed));
    
    json py_state_json = call_python_function("create_game_state", pArgs);
    Py_DECREF(pArgs);
    
    // Get legal actions from Python
    pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(py_state_json.dump().c_str()));
    json py_legal_actions = call_python_function("get_legal_actions", pArgs);
    Py_DECREF(pArgs);
    
    // Apply each legal action and compare results
    for (const auto& action_type : py_legal_actions) {
        // Create the action in C++
        Action cpp_action;
        if (action_type == 0) {
            cpp_action = Action::fold();
        } else if (action_type == 1) {
            cpp_action = Action::check();
        } else if (action_type == 2) {
            cpp_action = Action::call();
        } else if (action_type == 3) {
            // For raise, use a specific amount (e.g., min bet)
            float amount = cpp_state.get_min_bet();
            cpp_action = Action::raise(amount);
            
            // Apply the action in Python
            pArgs = PyTuple_New(3);
            PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(py_state_json.dump().c_str()));
            PyTuple_SetItem(pArgs, 1, PyLong_FromLong(action_type));
            PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(amount));
            json new_py_state_json = call_python_function("apply_action", pArgs);
            Py_DECREF(pArgs);
            
            // Apply the action in C++
            State new_cpp_state = cpp_state.apply_action(cpp_action);
            
            // Compare the new states
            REQUIRE(new_cpp_state.get_current_player() == new_py_state_json["current_player"]);
            REQUIRE(new_cpp_state.get_pot() == Approx(new_py_state_json["pot"].get<float>()));
            REQUIRE(new_cpp_state.get_min_bet() == Approx(new_py_state_json["min_bet"].get<float>()));
        } else {
            // Apply the action in Python without amount
            pArgs = PyTuple_New(2);
            PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(py_state_json.dump().c_str()));
            PyTuple_SetItem(pArgs, 1, PyLong_FromLong(action_type));
            json new_py_state_json = call_python_function("apply_action", pArgs);
            Py_DECREF(pArgs);
            
            // Apply the action in C++
            State new_cpp_state = cpp_state.apply_action(cpp_action);
            
            // Compare the new states
            REQUIRE(new_cpp_state.get_current_player() == new_py_state_json["current_player"]);
            REQUIRE(new_cpp_state.get_pot() == Approx(new_py_state_json["pot"].get<float>()));
        }
    }
}

// Test state encoding parity
TEST_CASE("State encoding parity", "[parity]") {
    // Parameters for state creation
    int n_players = 6;
    int button = 0;
    float sb = 1.0f;
    float bb = 2.0f;
    float stake = 200.0f;
    int seed = 42;
    
    // Create state with C++
    State cpp_state = State::from_seed(n_players, button, sb, bb, stake, seed);
    
    // Create state with Python
    PyObject* pArgs = PyTuple_New(6);
    PyTuple_SetItem(pArgs, 0, PyLong_FromLong(n_players));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(button));
    PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(sb));
    PyTuple_SetItem(pArgs, 3, PyFloat_FromDouble(bb));
    PyTuple_SetItem(pArgs, 4, PyFloat_FromDouble(stake));
    PyTuple_SetItem(pArgs, 5, PyLong_FromLong(seed));
    
    json py_state_json = call_python_function("create_game_state", pArgs);
    Py_DECREF(pArgs);
    
    // Test encoding for each player perspective
    for (int player_id = 0; player_id < n_players; ++player_id) {
        // Encode state in C++
        torch::Tensor cpp_encoded = encode_state(&cpp_state, player_id);
        std::vector<float> cpp_encoding(cpp_encoded.data_ptr<float>(),
                                      cpp_encoded.data_ptr<float>() + cpp_encoded.numel());
        
        // Encode state in Python
        pArgs = PyTuple_New(2);
        PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(py_state_json.dump().c_str()));
        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(player_id));
        json py_encoding = call_python_function("encode_state", pArgs);
        Py_DECREF(pArgs);
        
        // Compare encodings
        REQUIRE(cpp_encoding.size() == py_encoding.size());
        
        for (size_t i = 0; i < cpp_encoding.size(); ++i) {
            REQUIRE(cpp_encoding[i] == Approx(py_encoding[i].get<float>()).epsilon(0.001));
        }
    }
}

// Test game simulation parity
TEST_CASE("Full game simulation parity", "[parity]") {
    // Parameters for state creation
    int n_players = 6;
    int button = 0;
    float sb = 1.0f;
    float bb = 2.0f;
    float stake = 200.0f;
    int seed = 42;
    
    // Create initial states
    State cpp_state = State::from_seed(n_players, button, sb, bb, stake, seed);
    
    PyObject* pArgs = PyTuple_New(6);
    PyTuple_SetItem(pArgs, 0, PyLong_FromLong(n_players));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(button));
    PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(sb));
    PyTuple_SetItem(pArgs, 3, PyFloat_FromDouble(bb));
    PyTuple_SetItem(pArgs, 4, PyFloat_FromDouble(stake));
    PyTuple_SetItem(pArgs, 5, PyLong_FromLong(seed));
    
    json py_state_json = call_python_function("create_game_state", pArgs);
    Py_DECREF(pArgs);
    
    // Define a fixed action sequence for testing
    // We'll alternate between fold, check, call, and raise
    const std::vector<std::pair<ActionEnum, float>> action_sequence = {
        {ActionEnum::Call, 0.0f},   // Player 2 calls
        {ActionEnum::Fold, 0.0f},   // Player 3 folds
        {ActionEnum::Call, 0.0f},   // Player 4 calls
        {ActionEnum::Raise, 10.0f}, // Player 5 raises to 10
        {ActionEnum::Fold, 0.0f},   // Player 0 folds
        {ActionEnum::Call, 0.0f},   // Player 1 calls
        {ActionEnum::Call, 0.0f},   // Player 2 calls
        {ActionEnum::Call, 0.0f}    // Player 4 calls
    };
    
    // Apply actions to both states and compare
    for (const auto& [action_type, amount] : action_sequence) {
        // Skip if game is over
        if (cpp_state.is_final()) {
            break;
        }
        
        // Create the action in C++
        Action cpp_action;
        switch (action_type) {
            case ActionEnum::Fold:
                cpp_action = Action::fold();
                break;
            case ActionEnum::Check:
                cpp_action = Action::check();
                break;
            case ActionEnum::Call:
                cpp_action = Action::call();
                break;
            case ActionEnum::Raise:
                cpp_action = Action::raise(amount);
                break;
        }
        
        // Apply action in Python
        pArgs = PyTuple_New(3);
        PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(py_state_json.dump().c_str()));
        PyTuple_SetItem(pArgs, 1, PyLong_FromLong(static_cast<int>(action_type)));
        PyTuple_SetItem(pArgs, 2, PyFloat_FromDouble(amount));
        json new_py_state_json = call_python_function("apply_action", pArgs);
        Py_DECREF(pArgs);
        
        // Apply action in C++
        State new_cpp_state = cpp_state.apply_action(cpp_action);
        
        // Compare the new states
        REQUIRE(new_cpp_state.get_current_player() == new_py_state_json["current_player"]);
        REQUIRE(new_cpp_state.get_pot() == Approx(new_py_state_json["pot"].get<float>()));
        REQUIRE(new_cpp_state.get_min_bet() == Approx(new_py_state_json["min_bet"].get<float>()));
        
        // Update states for next iteration
        cpp_state = new_cpp_state;
        py_state_json = new_py_state_json;
    }
}
