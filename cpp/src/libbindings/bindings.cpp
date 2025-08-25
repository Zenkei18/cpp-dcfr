#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <memory>
#include <string>

#include "deepcfr/core/deep_cfr.h"
#include "deepcfr/model/model.h"
#include "deepcfr/pokers/pokers.h"

namespace py = pybind11;

namespace {
    // Helper function to convert Python objects to C++ types
    deepcfr::State py_state_to_cpp(py::object py_state) {
        // This function assumes py_state is a pokers.State object from the Python side
        // We need to extract its properties and create a C++ State object
        
        // Extract basic properties
        int n_players = py::len(py_state.attr("players_state"));
        int button = py_state.attr("button").cast<int>();
        float sb = 1.0f; // Default value
        float bb = 2.0f; // Default value
        float stake = 200.0f; // Default value
        int seed = 42; // Default seed
        
        // Create a base state
        auto cpp_state = deepcfr::State::from_seed(n_players, button, sb, bb, stake, seed);
        
        // For now, we'll leave it at this basic initialization
        // In a complete implementation, we'd copy all the state properties
        
        return cpp_state;
    }
    
    // Helper function to convert C++ action to Python action
    py::object cpp_action_to_py(const deepcfr::Action& action) {
        // This function would convert a C++ Action to a Python pokers.Action
        // For now, we just return a dictionary with the action properties
        py::dict result;
        result["action_type"] = static_cast<int>(action.get_type());
        result["amount"] = action.get_amount();
        return result;
    }
}

PYBIND11_MODULE(deepcfr_cpp, m) {
    m.doc() = "DeepCFR Poker AI C++ Extension";
    
    // Set global verbosity
    m.def("set_verbose", &deepcfr::set_verbose, "Set verbosity level");
    
    // Card class
    py::class_<deepcfr::Card>(m, "Card")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def_readwrite("suit", &deepcfr::Card::suit)
        .def_readwrite("rank", &deepcfr::Card::rank)
        .def("to_int", &deepcfr::Card::to_int)
        .def_static("from_int", &deepcfr::Card::from_int);
    
    // Enumerations
    py::enum_<deepcfr::ActionEnum>(m, "ActionEnum")
        .value("Fold", deepcfr::ActionEnum::Fold)
        .value("Check", deepcfr::ActionEnum::Check)
        .value("Call", deepcfr::ActionEnum::Call)
        .value("Raise", deepcfr::ActionEnum::Raise)
        .export_values();
    
    py::enum_<deepcfr::Stage>(m, "Stage")
        .value("Preflop", deepcfr::Stage::Preflop)
        .value("Flop", deepcfr::Stage::Flop)
        .value("Turn", deepcfr::Stage::Turn)
        .value("River", deepcfr::Stage::River)
        .value("Showdown", deepcfr::Stage::Showdown)
        .export_values();
    
    py::enum_<deepcfr::StateStatus>(m, "StateStatus")
        .value("Ok", deepcfr::StateStatus::Ok)
        .value("InvalidAction", deepcfr::StateStatus::InvalidAction)
        .value("GameOver", deepcfr::StateStatus::GameOver)
        .export_values();
    
    // PlayerState class
    py::class_<deepcfr::PlayerState>(m, "PlayerState")
        .def("get_hand_cards", &deepcfr::PlayerState::get_hand_cards)
        .def("get_stake", &deepcfr::PlayerState::get_stake)
        .def("get_bet_chips", &deepcfr::PlayerState::get_bet_chips)
        .def("get_pot_chips", &deepcfr::PlayerState::get_pot_chips)
        .def("is_active", &deepcfr::PlayerState::is_active)
        .def("get_reward", &deepcfr::PlayerState::get_reward);
    
    // Action class
    py::class_<deepcfr::Action>(m, "Action")
        .def_static("fold", &deepcfr::Action::fold)
        .def_static("check", &deepcfr::Action::check)
        .def_static("call", &deepcfr::Action::call)
        .def_static("raise", &deepcfr::Action::raise)
        .def("get_type", &deepcfr::Action::get_type)
        .def("get_amount", &deepcfr::Action::get_amount);
    
    // State class
    py::class_<deepcfr::State>(m, "State")
        .def_static("from_seed", &deepcfr::State::from_seed,
                 py::arg("n_players"), py::arg("button"), 
                 py::arg("sb"), py::arg("bb"), 
                 py::arg("stake"), py::arg("seed"))
        .def("clone", &deepcfr::State::clone)
        .def("get_current_player", &deepcfr::State::get_current_player)
        .def("get_pot", &deepcfr::State::get_pot)
        .def("get_min_bet", &deepcfr::State::get_min_bet)
        .def("get_button", &deepcfr::State::get_button)
        .def("get_stage", &deepcfr::State::get_stage)
        .def("get_status", &deepcfr::State::get_status)
        .def("is_final", &deepcfr::State::is_final)
        .def("get_player_state", &deepcfr::State::get_player_state)
        .def("get_community_cards", &deepcfr::State::get_community_cards)
        .def("get_legal_actions", &deepcfr::State::get_legal_actions)
        .def("apply_action", &deepcfr::State::apply_action);
    
    // PrioritizedMemory class
    py::class_<deepcfr::PrioritizedMemory>(m, "PrioritizedMemory")
        .def(py::init<size_t, float>(),
             py::arg("capacity"),
             py::arg("alpha") = 0.6f)
        .def("add", [](deepcfr::PrioritizedMemory& self, py::array_t<float> state, int action_id, float regret, float priority = -1.0f) {
            // Convert numpy array to torch::Tensor
            auto buffer = state.request();
            auto tensor = torch::from_blob(buffer.ptr, {static_cast<int64_t>(buffer.shape[0])}, torch::kFloat);
            
            self.add(tensor.clone(), action_id, regret, priority);
        }, "Add a new experience to memory",
           py::arg("state"),
           py::arg("action_id"),
           py::arg("regret"),
           py::arg("priority") = -1.0f)
        .def("sample", [](deepcfr::PrioritizedMemory& self, size_t batch_size, float beta = 0.4f) {
            auto [states, action_ids, regrets, indices, weights] = self.sample(batch_size, beta);
            
            // Convert torch::Tensor to numpy arrays
            auto states_shape = states.sizes().vec();
            std::vector<size_t> states_shape_size_t;
            for (const auto& s : states_shape) {
                states_shape_size_t.push_back(static_cast<size_t>(s));
            }
            
            auto states_array = py::array_t<float>(states_shape_size_t);
            std::memcpy(states_array.mutable_data(), states.data_ptr<float>(), 
                       states.numel() * sizeof(float));
            
            auto action_ids_array = py::array_t<int64_t>({action_ids.numel()});
            std::memcpy(action_ids_array.mutable_data(), action_ids.data_ptr<int64_t>(), 
                       action_ids.numel() * sizeof(int64_t));
            
            auto regrets_array = py::array_t<float>({regrets.numel()});
            std::memcpy(regrets_array.mutable_data(), regrets.data_ptr<float>(), 
                       regrets.numel() * sizeof(float));
            
            auto weights_array = py::array_t<float>({weights.numel()});
            std::memcpy(weights_array.mutable_data(), weights.data_ptr<float>(), 
                       weights.numel() * sizeof(float));
            
            return py::make_tuple(states_array, action_ids_array, regrets_array, indices, weights_array);
        }, "Sample a batch of experiences based on their priorities",
           py::arg("batch_size"),
           py::arg("beta") = 0.4f)
        .def("update_priority", &deepcfr::PrioritizedMemory::update_priority, 
             "Update the priority of an experience",
             py::arg("index"),
             py::arg("priority"))
        .def("__len__", &deepcfr::PrioritizedMemory::size)
        .def("empty", &deepcfr::PrioritizedMemory::empty);
    
    // AgentInterface class
    py::class_<deepcfr::AgentInterface, std::shared_ptr<deepcfr::AgentInterface>>(m, "AgentInterface")
        .def("choose_action", &deepcfr::AgentInterface::choose_action)
        .def("get_player_id", &deepcfr::AgentInterface::get_player_id);
    
    // RandomAgent class
    py::class_<deepcfr::RandomAgent, deepcfr::AgentInterface, std::shared_ptr<deepcfr::RandomAgent>>(m, "RandomAgent")
        .def(py::init<int>())
        .def("choose_action", &deepcfr::RandomAgent::choose_action)
        .def("get_player_id", &deepcfr::RandomAgent::get_player_id);
    
    // DeepCFRAgent class
    py::class_<deepcfr::DeepCFRAgent, deepcfr::AgentInterface, std::shared_ptr<deepcfr::DeepCFRAgent>>(m, "DeepCFRAgent")
        .def(py::init<int, int, int, const std::string&>(),
             py::arg("player_id"),
             py::arg("num_players") = 6,
             py::arg("num_actions") = 4,
             py::arg("device") = "cpu")
        .def("choose_action", &deepcfr::DeepCFRAgent::choose_action)
        .def("get_player_id", &deepcfr::DeepCFRAgent::get_player_id)
        .def("cfr_traverse", &deepcfr::DeepCFRAgent::cfr_traverse,
             py::arg("state"),
             py::arg("iteration"),
             py::arg("opponent_agents"),
             py::arg("depth") = 0)
        .def("train_advantage_network", &deepcfr::DeepCFRAgent::train_advantage_network)
        .def("train_strategy_network", &deepcfr::DeepCFRAgent::train_strategy_network)
        .def("save_model", &deepcfr::DeepCFRAgent::save_model)
        .def("load_model", &deepcfr::DeepCFRAgent::load_model)
        .def("get_legal_action_ids", &deepcfr::DeepCFRAgent::get_legal_action_ids)
        .def("action_id_to_pokers_action", &deepcfr::DeepCFRAgent::action_id_to_pokers_action);
    
    // Encode state function
    m.def("encode_state", [](const deepcfr::State& state, int player_id) {
        torch::Tensor encoded = deepcfr::encode_state(&state, player_id);
        
        // Convert to numpy array
        auto sizes = encoded.sizes();
        std::vector<size_t> shape;
        for (const auto& s : sizes) {
            shape.push_back(static_cast<size_t>(s));
        }
        
        auto array = py::array_t<float>(shape);
        std::memcpy(array.mutable_data(), encoded.data_ptr<float>(), 
                   encoded.numel() * sizeof(float));
        
        return array;
    }, "Encode a poker state as a feature vector",
       py::arg("state"),
       py::arg("player_id") = 0);
    
    // Module-level functions to convert between Python and C++ objects
    m.def("py_state_to_cpp", &py_state_to_cpp, "Convert Python pokers.State to C++ State");
    m.def("cpp_action_to_py", &cpp_action_to_py, "Convert C++ Action to Python pokers.Action");
}