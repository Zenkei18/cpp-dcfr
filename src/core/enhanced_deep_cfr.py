# src/core/enhanced_deep_cfr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import pokers as pkrs
from collections import deque
from typing import Optional, Tuple, Dict, Any

from src.core.model import encode_state, VERBOSE
from src.core.enhanced_model import create_model, TargetNormalizer
from src.core.deep_cfr import PrioritizedMemory
from src.utils.settings import STRICT_CHECKING
from src.utils.logging import log_game_error
from src.utils.config import Config, create_optimizer, create_scheduler, set_seed


class EnhancedDeepCFRAgent:
    """Enhanced Deep CFR Agent with modern training optimizations."""
    
    def __init__(self, config: Config, player_id: int = 0, num_players: int = 6, device: str = 'cpu'):
        self.config = config
        self.player_id = player_id
        self.num_players = num_players
        self.device = device
        
        # Set random seeds for reproducibility
        set_seed(config.seed, config.deterministic)
        
        # Define action types (Fold, Check/Call, Raise)
        self.num_actions = config.model.num_actions
        
        # Calculate input size based on state encoding
        input_size = 52 + 52 + 5 + 1 + num_players + num_players + num_players*4 + 1 + 4 + 5
        config.model.input_size = input_size
        
        # Create networks
        self.advantage_net = create_model(config.model, device)
        self.strategy_net = create_model(config.model, device)
        
        # Create optimizers with separate parameter groups for advantage and strategy
        self.advantage_optimizer = self._create_optimizer(
            self.advantage_net.parameters(), 
            config.training.advantage_lr
        )
        self.strategy_optimizer = self._create_optimizer(
            self.strategy_net.parameters(),
            config.training.strategy_lr
        )
        
        # Create learning rate schedulers
        self.advantage_scheduler = create_scheduler(self.advantage_optimizer, config.training)
        self.strategy_scheduler = create_scheduler(self.strategy_optimizer, config.training)
        
        # Mixed precision training
        self.use_amp = config.training.use_amp
        if self.use_amp:
            self.advantage_scaler = torch.cuda.amp.GradScaler(init_scale=config.training.amp_init_scale)
            self.strategy_scaler = torch.cuda.amp.GradScaler(init_scale=config.training.amp_init_scale)
        
        # Create memory buffers
        self.advantage_memory = PrioritizedMemory(
            config.training.memory_size, 
            config.memory.alpha
        )
        self.strategy_memory = deque(maxlen=config.training.memory_size)
        
        # Target normalization
        if config.training.normalize_targets:
            self.target_normalizer = TargetNormalizer(config.training.target_scaler)
        else:
            self.target_normalizer = None
        
        # Training statistics
        self.iteration_count = 0
        self.step_count = 0
        self.training_stats = {
            'advantage_loss': [],
            'strategy_loss': [],
            'grad_norms': {'advantage': [], 'strategy': []},
            'learning_rates': {'advantage': [], 'strategy': []},
            'steps_per_sec': [],
            'amp_scales': {'advantage': [], 'strategy': []} if self.use_amp else None
        }
        
        # Bet sizing bounds
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0
        
    def _create_optimizer(self, parameters, learning_rate: float) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.training.optimizer.lower() == "adamw":
            return optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer.lower() == "adam":
            return optim.Adam(
                parameters,
                lr=learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def _log_gradient_norms(self, model: nn.Module, prefix: str) -> float:
        """Log gradient norms and return total norm."""
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.norm().item()
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        if self.config.logging.log_grad_norms:
            self.training_stats['grad_norms'][prefix].append(total_norm)
            
        return total_norm
    
    def _log_learning_rates(self) -> None:
        """Log current learning rates."""
        if self.config.logging.log_lr:
            adv_lr = self.advantage_optimizer.param_groups[0]['lr']
            strat_lr = self.strategy_optimizer.param_groups[0]['lr']
            
            self.training_stats['learning_rates']['advantage'].append(adv_lr)
            self.training_stats['learning_rates']['strategy'].append(strat_lr)
    
    def _update_target_normalization(self, regrets: np.ndarray) -> None:
        """Update target normalization statistics."""
        if self.target_normalizer is not None and len(regrets) > 0:
            if self.step_count % self.config.training.update_scaler_freq == 0:
                self.target_normalizer.update(regrets)
    
    def get_normalized_targets(self, regrets: np.ndarray) -> np.ndarray:
        """Get normalized target values."""
        if self.target_normalizer is not None:
            return self.target_normalizer.normalize(regrets)
        return regrets
    
    # Copy the core CFR methods from the original implementation
    def action_type_to_pokers_action(self, action_type, state, bet_size_multiplier=None):
        """Convert action type and optional bet size to Pokers action."""
        # This is the same implementation as the original DeepCFRAgent
        # Copy the exact implementation from the original class
        try:
            if action_type == 0:  # Fold
                if pkrs.ActionEnum.Fold in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                if pkrs.ActionEnum.Check in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Check)
                if pkrs.ActionEnum.Call in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Call)
                return pkrs.Action(pkrs.ActionEnum.Fold)

            elif action_type == 1:  # Check/Call
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                elif pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Fold in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                return pkrs.Action(pkrs.ActionEnum.Check)

            elif action_type == 2:  # Raise
                if pkrs.ActionEnum.Raise not in state.legal_actions:
                    if pkrs.ActionEnum.Call in state.legal_actions: 
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    if pkrs.ActionEnum.Check in state.legal_actions: 
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    return pkrs.Action(pkrs.ActionEnum.Fold)

                player_state = state.players_state[state.current_player]
                current_bet = player_state.bet_chips
                available_stake = player_state.stake
                call_amount = max(0.0, state.min_bet - current_bet)
                min_raise_increment = 1.0

                if available_stake < call_amount + min_raise_increment:
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    else:
                        return pkrs.Action(pkrs.ActionEnum.Fold)

                remaining_stake_after_call = available_stake - call_amount
                pot_size = max(1.0, state.pot)

                if bet_size_multiplier is None:
                    bet_size_multiplier = 1.0
                else:
                    bet_size_multiplier = float(bet_size_multiplier)

                bet_size_multiplier = max(self.min_bet_size, min(self.max_bet_size, bet_size_multiplier))
                network_desired_additional_raise = pot_size * bet_size_multiplier

                chosen_additional_amount = min(network_desired_additional_raise, remaining_stake_after_call)
                chosen_additional_amount = max(chosen_additional_amount, min_raise_increment)

                if chosen_additional_amount > remaining_stake_after_call:
                    chosen_additional_amount = remaining_stake_after_call

                chosen_additional_amount = max(0.0, chosen_additional_amount)

                return pkrs.Action(pkrs.ActionEnum.Raise, chosen_additional_amount)

            else:
                if pkrs.ActionEnum.Call in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Call)
                if pkrs.ActionEnum.Check in state.legal_actions: 
                    return pkrs.Action(pkrs.ActionEnum.Check)
                return pkrs.Action(pkrs.ActionEnum.Fold)

        except Exception as e:
            if VERBOSE:
                print(f"EnhancedDeepCFRAgent ERROR in action conversion: {e}")
            return pkrs.Action(pkrs.ActionEnum.Fold)
    
    def get_legal_action_types(self, state):
        """Get the legal action types for the current state."""
        legal_action_types = []
        
        if pkrs.ActionEnum.Fold in state.legal_actions:
            legal_action_types.append(0)
        if pkrs.ActionEnum.Check in state.legal_actions or pkrs.ActionEnum.Call in state.legal_actions:
            legal_action_types.append(1)
        if pkrs.ActionEnum.Raise in state.legal_actions:
            legal_action_types.append(2)
        
        return legal_action_types
    
    def cfr_traverse(self, state, iteration, random_agents, depth=0):
        """CFR traversal with enhanced features."""
        # Use the same implementation as original but with enhanced logging
        max_depth = 1000
        if depth > max_depth:
            if VERBOSE:
                print(f"WARNING: Max recursion depth reached ({max_depth}). Returning zero value.")
            return 0
        
        if state.final_state:
            return state.players_state[self.player_id].reward
        
        current_player = state.current_player
        
        if current_player == self.player_id:
            legal_action_types = self.get_legal_action_types(state)
            
            if not legal_action_types:
                if VERBOSE:
                    print(f"WARNING: No legal actions found for player {current_player} at depth {depth}")
                return 0
                
            state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).to(self.device)
            
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        advantages, bet_size_pred = self.advantage_net(state_tensor.unsqueeze(0))
                else:
                    advantages, bet_size_pred = self.advantage_net(state_tensor.unsqueeze(0))
                
                advantages = advantages[0].cpu().numpy()
                bet_size_multiplier = bet_size_pred[0][0].item()
            
            # Use regret matching
            advantages_masked = np.zeros(self.num_actions)
            for a in legal_action_types:
                advantages_masked[a] = max(advantages[a], 0)
                
            if sum(advantages_masked) > 0:
                strategy = advantages_masked / sum(advantages_masked)
            else:
                strategy = np.zeros(self.num_actions)
                for a in legal_action_types:
                    strategy[a] = 1.0 / len(legal_action_types)
            
            # Traverse actions
            action_values = np.zeros(self.num_actions)
            for action_type in legal_action_types:
                try:
                    if action_type == 2:
                        pokers_action = self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
                    else:
                        pokers_action = self.action_type_to_pokers_action(action_type, state)
                    
                    new_state = state.apply_action(pokers_action)
                    
                    if new_state.status != pkrs.StateStatus.Ok:
                        log_file = log_game_error(state, pokers_action, f"State status not OK ({new_state.status})")
                        if STRICT_CHECKING:
                            raise ValueError(f"State status not OK ({new_state.status}). Details logged to {log_file}")
                        elif VERBOSE:
                            print(f"WARNING: Invalid action at depth {depth}. Details logged to {log_file}")
                        continue
                        
                    action_values[action_type] = self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
                except Exception as e:
                    if VERBOSE:
                        print(f"ERROR in traversal for action {action_type}: {e}")
                    action_values[action_type] = 0
                    if STRICT_CHECKING:
                        raise
            
            # Compute regrets and add to memory
            ev = sum(strategy[a] * action_values[a] for a in legal_action_types)
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            
            regrets = []
            for action_type in legal_action_types:
                regret = action_values[action_type] - ev
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -10.0, 10.0)
                scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0
                weighted_regret = clipped_regret * scale_factor
                regrets.append(weighted_regret)
                
                priority = abs(weighted_regret) + 0.01
                
                if action_type == 2:
                    self.advantage_memory.add(
                        (encode_state(state, self.player_id), 
                         np.zeros(20),
                         action_type, 
                         bet_size_multiplier, 
                         weighted_regret),
                        priority
                    )
                else:
                    self.advantage_memory.add(
                        (encode_state(state, self.player_id),
                         np.zeros(20),
                         action_type, 
                         0.0,
                         weighted_regret),
                        priority
                    )
            
            # Update target normalization
            if regrets:
                self._update_target_normalization(np.array(regrets))
            
            # Add to strategy memory
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_types:
                strategy_full[a] = strategy[a]
            
            self.strategy_memory.append((
                encode_state(state, self.player_id),
                np.zeros(20),
                strategy_full,
                bet_size_multiplier if 2 in legal_action_types else 0.0,
                iteration
            ))
            
            return ev
            
        else:
            try:
                action = random_agents[current_player].choose_action(state)
                new_state = state.apply_action(action)
                
                if new_state.status != pkrs.StateStatus.Ok:
                    log_file = log_game_error(state, action, f"State status not OK ({new_state.status})")
                    if STRICT_CHECKING:
                        raise ValueError(f"State status not OK. Details logged to {log_file}")
                    if VERBOSE:
                        print(f"WARNING: Random agent made invalid action. Details logged to {log_file}")
                    return 0
                    
                return self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
            except Exception as e:
                if VERBOSE:
                    print(f"ERROR in random agent traversal: {e}")
                if STRICT_CHECKING:
                    raise
                return 0
    
    def train_advantage_network(self, batch_size: Optional[int] = None, epochs: Optional[int] = None) -> float:
        """Train the advantage network with enhanced optimizations."""
        if batch_size is None:
            batch_size = self.config.training.batch_size
        if epochs is None:
            epochs = self.config.training.epochs_per_update
            
        if len(self.advantage_memory) < batch_size:
            return 0
        
        self.advantage_net.train()
        total_loss = 0
        start_time = time.time()
        
        # Calculate current beta for importance sampling
        progress = min(1.0, self.iteration_count / 10000)
        beta = self.config.memory.beta_start + progress * (self.config.memory.beta_end - self.config.memory.beta_start)
        
        for epoch in range(epochs):
            # Sample batch
            batch, indices, weights = self.advantage_memory.sample(batch_size, beta=beta)
            states, opponent_features, action_types, bet_sizes, regrets = zip(*batch)
            
            # Convert to tensors
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            action_type_tensors = torch.LongTensor(np.array(action_types)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            
            # Normalize targets if enabled
            regrets_np = np.array(regrets)
            normalized_regrets = self.get_normalized_targets(regrets_np)
            regret_tensors = torch.FloatTensor(normalized_regrets).to(self.device)
            weight_tensors = torch.FloatTensor(weights).to(self.device)
            
            # Forward pass with AMP if enabled
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    action_advantages, bet_size_preds = self.advantage_net(state_tensors, opponent_feature_tensors)
                    predicted_regrets = action_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
                    action_loss = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
                    weighted_action_loss = (action_loss * weight_tensors).mean()
                    
                    # Bet sizing loss for raise actions
                    raise_mask = (action_type_tensors == 2)
                    if torch.any(raise_mask):
                        all_bet_losses = F.smooth_l1_loss(bet_size_preds, bet_size_tensors, reduction='none')
                        masked_bet_losses = all_bet_losses * raise_mask.float().unsqueeze(1)
                        raise_count = raise_mask.sum().item()
                        if raise_count > 0:
                            weighted_bet_size_loss = (masked_bet_losses.squeeze() * weight_tensors).sum() / raise_count
                            combined_loss = weighted_action_loss + 0.5 * weighted_bet_size_loss
                        else:
                            combined_loss = weighted_action_loss
                    else:
                        combined_loss = weighted_action_loss
                
                # Backward pass with AMP
                self.advantage_optimizer.zero_grad()
                self.advantage_scaler.scale(combined_loss).backward()
                
                # Gradient clipping
                self.advantage_scaler.unscale_(self.advantage_optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.advantage_net.parameters(), 
                    max_norm=self.config.training.gradient_clip_norm
                )
                
                self.advantage_scaler.step(self.advantage_optimizer)
                self.advantage_scaler.update()
                
            else:
                # Standard forward pass
                action_advantages, bet_size_preds = self.advantage_net(state_tensors, opponent_feature_tensors)
                predicted_regrets = action_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
                action_loss = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
                weighted_action_loss = (action_loss * weight_tensors).mean()
                
                # Bet sizing loss
                raise_mask = (action_type_tensors == 2)
                if torch.any(raise_mask):
                    all_bet_losses = F.smooth_l1_loss(bet_size_preds, bet_size_tensors, reduction='none')
                    masked_bet_losses = all_bet_losses * raise_mask.float().unsqueeze(1)
                    raise_count = raise_mask.sum().item()
                    if raise_count > 0:
                        weighted_bet_size_loss = (masked_bet_losses.squeeze() * weight_tensors).sum() / raise_count
                        combined_loss = weighted_action_loss + 0.5 * weighted_bet_size_loss
                    else:
                        combined_loss = weighted_action_loss
                else:
                    combined_loss = weighted_action_loss
                
                # Standard backward pass
                self.advantage_optimizer.zero_grad()
                combined_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.advantage_net.parameters(), 
                    max_norm=self.config.training.gradient_clip_norm
                )
                
                self.advantage_optimizer.step()
            
            # Update priorities
            with torch.no_grad():
                new_action_errors = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
                if torch.any(raise_mask):
                    new_bet_errors = torch.zeros_like(new_action_errors)
                    raise_indices = torch.where(raise_mask)[0]
                    for i in raise_indices:
                        new_bet_errors[i] = F.smooth_l1_loss(
                            bet_size_preds[i], bet_size_tensors[i], reduction='mean'
                        )
                    combined_errors = new_action_errors + 0.5 * new_bet_errors
                else:
                    combined_errors = new_action_errors
                
                combined_errors_np = combined_errors.cpu().numpy()
                for i, idx in enumerate(indices):
                    self.advantage_memory.update_priority(idx, combined_errors_np[i] + 0.01)
            
            # Log gradient norms
            if self.config.logging.log_grad_norms:
                self._log_gradient_norms(self.advantage_net, 'advantage')
            
            total_loss += combined_loss.item()
        
        # Step learning rate scheduler
        if self.advantage_scheduler is not None:
            self.advantage_scheduler.step()
        
        # Log training statistics
        avg_loss = total_loss / epochs
        self.training_stats['advantage_loss'].append(avg_loss)
        
        if self.config.logging.log_steps_per_sec:
            elapsed_time = time.time() - start_time
            steps_per_sec = (batch_size * epochs) / elapsed_time
            self.training_stats['steps_per_sec'].append(steps_per_sec)
        
        if self.use_amp and self.config.logging.log_amp_scale:
            self.training_stats['amp_scales']['advantage'].append(
                self.advantage_scaler.get_scale()
            )
        
        self.step_count += 1
        return avg_loss
    
    def train_strategy_network(self, batch_size: Optional[int] = None, epochs: Optional[int] = None) -> float:
        """Train the strategy network with enhanced optimizations."""
        if batch_size is None:
            batch_size = self.config.training.batch_size
        if epochs is None:
            epochs = self.config.training.epochs_per_update
            
        if len(self.strategy_memory) < batch_size:
            return 0
        
        self.strategy_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch
            batch = random.sample(self.strategy_memory, batch_size)
            states, opponent_features, strategies, bet_sizes, iterations = zip(*batch)
            
            # Convert to tensors
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            
            # Weight samples by iteration (Linear CFR)
            weights = iteration_tensors / torch.sum(iteration_tensors)
            
            # Forward pass with AMP if enabled
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    action_logits, bet_size_preds = self.strategy_net(state_tensors)
                    predicted_strategies = F.softmax(action_logits, dim=1)
                    
                    action_loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1))
                    
                    # Bet size loss
                    raise_mask = (strategy_tensors[:, 2] > 0)
                    if raise_mask.sum() > 0:
                        raise_indices = torch.nonzero(raise_mask).squeeze(1)
                        raise_bet_preds = bet_size_preds[raise_indices]
                        raise_bet_targets = bet_size_tensors[raise_indices]
                        raise_weights = weights[raise_indices]
                        
                        bet_size_loss = F.smooth_l1_loss(raise_bet_preds, raise_bet_targets, reduction='none')
                        weighted_bet_size_loss = torch.sum(raise_weights * bet_size_loss.squeeze())
                        combined_loss = action_loss + 0.5 * weighted_bet_size_loss
                    else:
                        combined_loss = action_loss
                
                # Backward pass with AMP
                self.strategy_optimizer.zero_grad()
                self.strategy_scaler.scale(combined_loss).backward()
                
                self.strategy_scaler.unscale_(self.strategy_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.strategy_net.parameters(), 
                    max_norm=self.config.training.gradient_clip_norm
                )
                
                self.strategy_scaler.step(self.strategy_optimizer)
                self.strategy_scaler.update()
            
            else:
                # Standard forward pass
                action_logits, bet_size_preds = self.strategy_net(state_tensors)
                predicted_strategies = F.softmax(action_logits, dim=1)
                
                action_loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1))
                
                # Bet size loss
                raise_mask = (strategy_tensors[:, 2] > 0)
                if raise_mask.sum() > 0:
                    raise_indices = torch.nonzero(raise_mask).squeeze(1)
                    raise_bet_preds = bet_size_preds[raise_indices]
                    raise_bet_targets = bet_size_tensors[raise_indices]
                    raise_weights = weights[raise_indices]
                    
                    bet_size_loss = F.smooth_l1_loss(raise_bet_preds, raise_bet_targets, reduction='none')
                    weighted_bet_size_loss = torch.sum(raise_weights * bet_size_loss.squeeze())
                    combined_loss = action_loss + 0.5 * weighted_bet_size_loss
                else:
                    combined_loss = action_loss
                
                # Standard backward pass
                self.strategy_optimizer.zero_grad()
                combined_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.strategy_net.parameters(), 
                    max_norm=self.config.training.gradient_clip_norm
                )
                
                self.strategy_optimizer.step()
            
            # Log gradient norms
            if self.config.logging.log_grad_norms:
                self._log_gradient_norms(self.strategy_net, 'strategy')
            
            total_loss += combined_loss.item()
        
        # Step learning rate scheduler
        if self.strategy_scheduler is not None:
            self.strategy_scheduler.step()
        
        # Log training statistics
        avg_loss = total_loss / epochs
        self.training_stats['strategy_loss'].append(avg_loss)
        
        if self.use_amp and self.config.logging.log_amp_scale:
            self.training_stats['amp_scales']['strategy'].append(
                self.strategy_scaler.get_scale()
            )
        
        return avg_loss
    
    def choose_action(self, state):
        """Choose an action for the given state during actual play."""
        legal_action_types = self.get_legal_action_types(state)
        
        if not legal_action_types:
            if pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
            elif pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            else:
                return pkrs.Action(pkrs.ActionEnum.Fold)
            
        state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, bet_size_pred = self.strategy_net(state_tensor)
            else:
                logits, bet_size_pred = self.strategy_net(state_tensor)
                
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()
        
        # Filter to only legal actions
        legal_probs = np.array([probs[a] for a in legal_action_types])
        if np.sum(legal_probs) > 0:
            legal_probs = legal_probs / np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_action_types)) / len(legal_action_types)
        
        # Choose action
        action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
        action_type = legal_action_types[action_idx]
        
        if action_type == 2:
            return self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
        else:
            return self.action_type_to_pokers_action(action_type, state)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = self.training_stats.copy()
        
        # Add target normalization stats if available
        if self.target_normalizer is not None and self.config.logging.log_target_stats:
            stats['target_normalization'] = self.target_normalizer.get_stats()
        
        # Add current learning rates
        stats['current_lr'] = {
            'advantage': self.advantage_optimizer.param_groups[0]['lr'],
            'strategy': self.strategy_optimizer.param_groups[0]['lr']
        }
        
        return stats
    
    def save_model(self, path_prefix: str) -> None:
        """Save the enhanced model."""
        save_dict = {
            'iteration': self.iteration_count,
            'step_count': self.step_count,
            'config': self.config,
            'advantage_net': self.advantage_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict(),
            'advantage_optimizer': self.advantage_optimizer.state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'training_stats': self.training_stats
        }
        
        if self.advantage_scheduler is not None:
            save_dict['advantage_scheduler'] = self.advantage_scheduler.state_dict()
        if self.strategy_scheduler is not None:
            save_dict['strategy_scheduler'] = self.strategy_scheduler.state_dict()
        
        if self.use_amp:
            save_dict['advantage_scaler'] = self.advantage_scaler.state_dict()
            save_dict['strategy_scaler'] = self.strategy_scaler.state_dict()
        
        if self.target_normalizer is not None:
            save_dict['target_normalizer'] = {
                'scaler_type': self.target_normalizer.scaler_type,
                'mean': self.target_normalizer.mean,
                'std': self.target_normalizer.std,
                'median': self.target_normalizer.median,
                'iqr': self.target_normalizer.iqr,
                'count': self.target_normalizer.count
            }
        
        torch.save(save_dict, f"{path_prefix}_enhanced_iter_{self.iteration_count}.pt")
    
    def load_model(self, path: str) -> None:
        """Load the enhanced model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.iteration_count = checkpoint.get('iteration', 0)
        self.step_count = checkpoint.get('step_count', 0)
        
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
        
        if 'advantage_optimizer' in checkpoint:
            self.advantage_optimizer.load_state_dict(checkpoint['advantage_optimizer'])
        if 'strategy_optimizer' in checkpoint:
            self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer'])
        
        if self.advantage_scheduler is not None and 'advantage_scheduler' in checkpoint:
            self.advantage_scheduler.load_state_dict(checkpoint['advantage_scheduler'])
        if self.strategy_scheduler is not None and 'strategy_scheduler' in checkpoint:
            self.strategy_scheduler.load_state_dict(checkpoint['strategy_scheduler'])
        
        if self.use_amp:
            if 'advantage_scaler' in checkpoint:
                self.advantage_scaler.load_state_dict(checkpoint['advantage_scaler'])
            if 'strategy_scaler' in checkpoint:
                self.strategy_scaler.load_state_dict(checkpoint['strategy_scaler'])
        
        if 'target_normalizer' in checkpoint and self.target_normalizer is not None:
            tn_data = checkpoint['target_normalizer']
            self.target_normalizer.scaler_type = tn_data['scaler_type']
            self.target_normalizer.mean = tn_data['mean']
            self.target_normalizer.std = tn_data['std']
            self.target_normalizer.median = tn_data['median']
            self.target_normalizer.iqr = tn_data['iqr']
            self.target_normalizer.count = tn_data['count']
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']