#include "deepcfr/pokers/pokers.h"
#include <stdexcept>

namespace deepcfr {

// PlayerState implementation

PlayerState::PlayerState(std::unique_ptr<pokers::PlayerState> player_state_handle)
    : handle_(std::move(player_state_handle)) {}

std::array<Card, 2> PlayerState::get_hand_cards() const {
    std::array<int, 2> card_values{};
    pokers::player_state_get_hand_cards(*handle_, &card_values);
    
    std::array<Card, 2> cards;
    cards[0] = Card::from_int(card_values[0]);
    cards[1] = Card::from_int(card_values[1]);
    
    return cards;
}

float PlayerState::get_stake() const {
    return pokers::player_state_get_stake(*handle_);
}

float PlayerState::get_bet_chips() const {
    return pokers::player_state_get_bet_chips(*handle_);
}

float PlayerState::get_pot_chips() const {
    return pokers::player_state_get_pot_chips(*handle_);
}

bool PlayerState::is_active() const {
    return pokers::player_state_is_active(*handle_);
}

float PlayerState::get_reward() const {
    return pokers::player_state_get_reward(*handle_);
}

// Action implementation

Action Action::fold() {
    return Action(pokers::action_create_fold());
}

Action Action::check() {
    return Action(pokers::action_create_check());
}

Action Action::call() {
    return Action(pokers::action_create_call());
}

Action Action::raise(float amount) {
    return Action(pokers::action_create_raise(amount));
}

Action::Action(std::unique_ptr<pokers::ActionHandle> handle)
    : handle_(std::move(handle)) {}

ActionEnum Action::get_type() const {
    auto type = pokers::action_get_type(*handle_);
    switch (type) {
        case pokers::ActionEnum::Fold:
            return ActionEnum::Fold;
        case pokers::ActionEnum::Check:
            return ActionEnum::Check;
        case pokers::ActionEnum::Call:
            return ActionEnum::Call;
        case pokers::ActionEnum::Raise:
            return ActionEnum::Raise;
        default:
            throw std::runtime_error("Unknown action type");
    }
}

float Action::get_amount() const {
    return pokers::action_get_amount(*handle_);
}

// State implementation

State State::from_seed(int n_players, int button, float sb, float bb, float stake, int seed) {
    return State(pokers::state_create(n_players, button, sb, bb, stake, seed));
}

State::State(std::unique_ptr<pokers::StateHandle> handle)
    : handle_(std::move(handle)) {}

State State::clone() const {
    return State(pokers::state_clone(*handle_));
}

int State::get_current_player() const {
    return pokers::state_get_current_player(*handle_);
}

float State::get_pot() const {
    return pokers::state_get_pot(*handle_);
}

float State::get_min_bet() const {
    return pokers::state_get_min_bet(*handle_);
}

int State::get_button() const {
    return pokers::state_get_button(*handle_);
}

Stage State::get_stage() const {
    auto stage = pokers::state_get_stage(*handle_);
    switch (stage) {
        case pokers::Stage::Preflop:
            return Stage::Preflop;
        case pokers::Stage::Flop:
            return Stage::Flop;
        case pokers::Stage::Turn:
            return Stage::Turn;
        case pokers::Stage::River:
            return Stage::River;
        case pokers::Stage::Showdown:
            return Stage::Showdown;
        default:
            throw std::runtime_error("Unknown stage");
    }
}

StateStatus State::get_status() const {
    auto status = pokers::state_get_status(*handle_);
    switch (status) {
        case pokers::StateStatus::Ok:
            return StateStatus::Ok;
        case pokers::StateStatus::InvalidAction:
            return StateStatus::InvalidAction;
        case pokers::StateStatus::GameOver:
            return StateStatus::GameOver;
        default:
            throw std::runtime_error("Unknown status");
    }
}

bool State::is_final() const {
    return pokers::state_is_final(*handle_);
}

PlayerState State::get_player_state(int player_id) const {
    return PlayerState(pokers::state_get_player_state(*handle_, player_id));
}

std::vector<Card> State::get_community_cards() const {
    int count = pokers::state_get_community_cards_count(*handle_);
    std::vector<int> card_values(count);
    pokers::state_get_community_cards(*handle_, card_values.data());
    
    std::vector<Card> cards;
    cards.reserve(count);
    for (int value : card_values) {
        cards.push_back(Card::from_int(value));
    }
    
    return cards;
}

std::vector<ActionEnum> State::get_legal_actions() const {
    std::vector<pokers::ActionEnum> rust_actions = pokers::state_get_legal_actions(*handle_);
    
    std::vector<ActionEnum> actions;
    actions.reserve(rust_actions.size());
    
    for (auto action : rust_actions) {
        switch (action) {
            case pokers::ActionEnum::Fold:
                actions.push_back(ActionEnum::Fold);
                break;
            case pokers::ActionEnum::Check:
                actions.push_back(ActionEnum::Check);
                break;
            case pokers::ActionEnum::Call:
                actions.push_back(ActionEnum::Call);
                break;
            case pokers::ActionEnum::Raise:
                actions.push_back(ActionEnum::Raise);
                break;
            default:
                throw std::runtime_error("Unknown action");
        }
    }
    
    return actions;
}

State State::apply_action(const Action& action) const {
    return State(pokers::state_apply_action(*handle_, action.raw_handle()));
}

} // namespace deepcfr
