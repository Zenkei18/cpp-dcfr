# src/core/enhanced_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from src.utils.config import ModelConfig


class ResidualBlock(nn.Module):
    """Residual block with normalization and activation."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.0, use_layer_norm: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        out = self.norm1(x)
        out = F.relu(out)
        out = self.linear1(out)
        out = self.dropout(out)
        
        out = self.norm2(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.dropout(out)
        
        # Residual connection
        out = out + identity
        return out


class EnhancedPokerNetwork(nn.Module):
    """Enhanced poker network with modern architecture improvements."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_actions = config.num_actions
        
        # Input projection
        self.input_proj = nn.Linear(config.input_size, config.hidden_size)
        self.input_dropout = nn.Dropout(config.dropout)
        
        if config.use_layer_norm:
            self.input_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.input_norm = nn.Identity()
        
        # Shared feature extraction layers
        if config.use_residuals:
            # Use residual blocks for better gradient flow
            self.feature_blocks = nn.ModuleList([
                ResidualBlock(config.hidden_size, config.dropout, config.use_layer_norm)
                for _ in range(3)  # 3 residual blocks
            ])
        else:
            # Traditional feedforward layers
            layers = []
            for i in range(3):
                layers.extend([
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.LayerNorm(config.hidden_size) if config.use_layer_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
                ])
            self.feature_blocks = nn.Sequential(*layers)
        
        # Action type prediction head (advantage head)
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2) if config.use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_size // 2, config.num_actions)
        )
        
        # Continuous bet sizing prediction head
        self.sizing_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2) if config.use_layer_norm else nn.Identity(),
            nn.Tanh(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()  # Output between 0-1
        )
        
        # Initialize weights with better initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU networks
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, opponent_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the enhanced network.
        
        Args:
            x: The state representation tensor
            opponent_features: Optional opponent modeling features (ignored in base class)
            
        Returns:
            Tuple of (action_logits, bet_size_prediction)
        """
        # Input processing
        features = self.input_proj(x)
        features = self.input_norm(features)
        features = F.relu(features)
        features = self.input_dropout(features)
        
        # Feature extraction
        if self.config.use_residuals:
            for block in self.feature_blocks:
                features = block(features)
        else:
            features = self.feature_blocks(features)
        
        # Output heads
        action_logits = self.action_head(features)
        bet_size = 0.1 + 2.9 * self.sizing_head(features)  # Scale to [0.1, 3.0]
        
        return action_logits, bet_size


class TargetNormalizer:
    """Target normalizer for advantage values using robust statistics."""
    
    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        self.mean = 0.0
        self.std = 1.0
        self.median = 0.0
        self.iqr = 1.0
        self.count = 0
        
    def update(self, values: np.ndarray) -> None:
        """Update normalization statistics with new values."""
        if len(values) == 0:
            return
            
        self.count += len(values)
        
        if self.scaler_type == "robust":
            # Use robust statistics (median, IQR)
            self.median = np.median(values)
            q75, q25 = np.percentile(values, [75, 25])
            self.iqr = max(q75 - q25, 1e-8)  # Avoid division by zero
        else:  # z-score
            # Use mean and standard deviation
            self.mean = np.mean(values)
            self.std = max(np.std(values), 1e-8)  # Avoid division by zero
    
    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize values using current statistics."""
        if self.scaler_type == "robust":
            return (values - self.median) / self.iqr
        else:  # z-score
            return (values - self.mean) / self.std
    
    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """Denormalize values back to original scale."""
        if self.scaler_type == "robust":
            return values * self.iqr + self.median
        else:  # z-score
            return values * self.std + self.mean
    
    def get_stats(self) -> dict:
        """Get current normalization statistics."""
        if self.scaler_type == "robust":
            return {"median": self.median, "iqr": self.iqr, "count": self.count}
        else:
            return {"mean": self.mean, "std": self.std, "count": self.count}


def create_model(config: ModelConfig, device: str = 'cpu') -> nn.Module:
    """
    Factory function to create model based on configuration.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Configured model
    """
    if config.name == "EnhancedPokerNetwork":
        model = EnhancedPokerNetwork(config)
    elif config.name == "PokerNetwork":
        # Import the original model for baseline comparison
        from src.core.model import PokerNetwork
        model = PokerNetwork(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_actions=config.num_actions
        )
    else:
        raise ValueError(f"Unknown model name: {config.name}")
    
    return model.to(device)