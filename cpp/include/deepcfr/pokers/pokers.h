#ifndef DEEPCFR_POKERS_POKERS_H_
#define DEEPCFR_POKERS_POKERS_H_

#include <memory>
#include <vector>
#include <array>
#include <optional>

#include "pokers-ffi/lib.rs.h"

namespace deepcfr {

// Forward declarations
class PlayerState;
class Action;
class State;

/**
 * @brief Represents a card in a standard deck
 */
struct Card {
    int suit;  // 0-3: clubs, diamonds, hearts, spades
    int rank;  // 0-12: 2-Ace
    
    /**
     * @brief Convert to integer representation
     * 
     * @return int A single integer representing this card
     */
    int to_int() const { return suit * 13 + rank; }
    
    /**
     * @brief Create a card from an integer representation
     * 
     * @param value Integer value
     * @return Card The corresponding card
     */
    static Card from_int(int value) {
        return {value / 13, value % 13};
    }
};

/**
 * @brief Enumeration of poker game stages
 */
enum class Stage {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
    Showdown = 4
};

/**
 * @brief Enumeration of possible actions in poker
 */
enum class ActionEnum {
    Fold = 0,
    Check = 1,
    Call = 2,
    Raise = 3
};

/**
 * @brief Enumeration of state status codes
 */
enum class StateStatus {
    Ok = 0,
    InvalidAction = 1,
    GameOver = 2
};

/**
 * @brief Represents a player's state in a poker game
 */
class PlayerState {
public:
    /**
     * @brief Construct a new Player State object
     * 
     * @param player_state_handle Handle to Rust player state
     */
    explicit PlayerState(std::unique_ptr<pokers::PlayerState> player_state_handle);
    
    /**
     * @brief Get the player's hand cards
     * 
     * @return std::array<Card, 2> Array of two cards
     */
    std::array<Card, 2> get_hand_cards() const;
    
    /**
     * @brief Get the player's current stake
     * 
     * @return float Current stake
     */
    float get_stake() const;
    
    /**
     * @brief Get the player's current bet
     * 
     * @return float Current bet
     */
    float get_bet_chips() const;
    
    /**
     * @brief Get the player's pot chips
     * 
     * @return float Pot chips
     */
    float get_pot_chips() const;
    
    /**
     * @brief Check if the player is active
     * 
     * @return true if active, false otherwise
     */
    bool is_active() const;
    
    /**
     * @brief Get the player's reward
     * 
     * @return float Reward
     */
    float get_reward() const;
    
private:
    std::unique_ptr<pokers::PlayerState> handle_;
};

/**
 * @brief Represents a poker action
 */
class Action {
public:
    /**
     * @brief Create a fold action
     * 
     * @return Action Fold action
     */
    static Action fold();
    
    /**
     * @brief Create a check action
     * 
     * @return Action Check action
     */
    static Action check();
    
    /**
     * @brief Create a call action
     * 
     * @return Action Call action
     */
    static Action call();
    
    /**
     * @brief Create a raise action
     * 
     * @param amount Amount to raise to
     * @return Action Raise action
     */
    static Action raise(float amount);
    
    /**
     * @brief Get the action type
     * 
     * @return ActionEnum The type of action
     */
    ActionEnum get_type() const;
    
    /**
     * @brief Get the action amount
     * 
     * @return float The action amount (for raise actions)
     */
    float get_amount() const;
    
    /**
     * @brief Get the raw handle to the Rust action
     */
    const pokers::ActionHandle& raw_handle() const { return *handle_; }
    
private:
    explicit Action(std::unique_ptr<pokers::ActionHandle> handle);
    std::unique_ptr<pokers::ActionHandle> handle_;
};

/**
 * @brief Represents the state of a poker game
 */
class State {
public:
    /**
     * @brief Create a new poker game state
     * 
     * @param n_players Number of players
     * @param button Button position
     * @param sb Small blind amount
     * @param bb Big blind amount
     * @param stake Initial stake for each player
     * @param seed Random seed
     * @return State New game state
     */
    static State from_seed(int n_players, int button, float sb, float bb, float stake, int seed);
    
    /**
     * @brief Clone the state
     * 
     * @return State A copy of this state
     */
    State clone() const;
    
    /**
     * @brief Get the current player
     * 
     * @return int Current player ID
     */
    int get_current_player() const;
    
    /**
     * @brief Get the pot size
     * 
     * @return float Pot size
     */
    float get_pot() const;
    
    /**
     * @brief Get the minimum bet
     * 
     * @return float Minimum bet
     */
    float get_min_bet() const;
    
    /**
     * @brief Get the button position
     * 
     * @return int Button position
     */
    int get_button() const;
    
    /**
     * @brief Get the current game stage
     * 
     * @return Stage Game stage
     */
    Stage get_stage() const;
    
    /**
     * @brief Get the state status
     * 
     * @return StateStatus Status code
     */
    StateStatus get_status() const;
    
    /**
     * @brief Check if this is a final state
     * 
     * @return true if final, false otherwise
     */
    bool is_final() const;
    
    /**
     * @brief Get a player's state
     * 
     * @param player_id Player ID
     * @return PlayerState The player's state
     */
    PlayerState get_player_state(int player_id) const;
    
    /**
     * @brief Get the community cards
     * 
     * @return std::vector<Card> Vector of community cards
     */
    std::vector<Card> get_community_cards() const;
    
    /**
     * @brief Get the legal actions
     * 
     * @return std::vector<ActionEnum> Vector of legal actions
     */
    std::vector<ActionEnum> get_legal_actions() const;
    
    /**
     * @brief Apply an action to the state
     * 
     * @param action Action to apply
     * @return State New state after applying the action
     */
    State apply_action(const Action& action) const;
    
    /**
     * @brief Get the raw handle to the Rust state
     */
    const pokers::StateHandle& raw_handle() const { return *handle_; }
    
private:
    explicit State(std::unique_ptr<pokers::StateHandle> handle);
    std::unique_ptr<pokers::StateHandle> handle_;
};

} // namespace deepcfr

#endif // DEEPCFR_POKERS_POKERS_H_
