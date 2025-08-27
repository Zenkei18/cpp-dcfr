# src/utils/config.py
import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import torch


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    name: str = "PokerNetwork"
    input_size: int = 156
    hidden_size: int = 256
    num_actions: int = 3
    dropout: float = 0.0
    use_layer_norm: bool = False
    use_batch_norm: bool = False
    use_residuals: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    optimizer: str = "adam"
    advantage_lr: float = 1e-6
    strategy_lr: float = 5e-5
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    scheduler: str = "none"
    warmup_steps: int = 0
    max_steps: int = 10000
    min_lr_ratio: float = 0.1
    max_lr_multiplier: float = 3.0
    
    batch_size: int = 128
    epochs_per_update: int = 3
    memory_size: int = 300000
    
    gradient_clip_norm: float = 0.5
    use_amp: bool = False
    amp_init_scale: float = 65536.0
    
    normalize_targets: bool = False
    target_scaler: str = "none"
    update_scaler_freq: int = 100


@dataclass
class MemoryConfig:
    """Configuration for memory buffer."""
    prioritized: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_grad_norms: bool = False
    log_lr: bool = False
    log_steps_per_sec: bool = False
    log_amp_scale: bool = False
    log_target_stats: bool = False
    tensorboard: bool = True


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    training: TrainingConfig
    memory: MemoryConfig
    logging: LoggingConfig
    seed: int = 42
    deterministic: bool = False


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Parsed configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects from dictionaries
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    memory_config = MemoryConfig(**config_dict.get('memory', {}))
    logging_config = LoggingConfig(**config_dict.get('logging', {}))
    
    return Config(
        model=model_config,
        training=training_config,
        memory=memory_config,
        logging=logging_config,
        seed=config_dict.get('seed', 42),
        deterministic=config_dict.get('deterministic', False)
    )


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        config_path: Output path for YAML file
    """
    config_dict = asdict(config)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def create_optimizer(model_parameters, config: TrainingConfig) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model_parameters: Model parameters to optimize
        config: Training configuration
        
    Returns:
        Configured optimizer
    """
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            model_parameters,
            lr=config.advantage_lr,  # Will be overridden per parameter group
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )
    elif config.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model_parameters,
            lr=config.advantage_lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Training configuration
        
    Returns:
        Configured scheduler or None
    """
    if config.scheduler.lower() == "none":
        return None
    elif config.scheduler.lower() == "linear_warmup":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
        )
    elif config.scheduler.lower() == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.max_steps - config.warmup_steps,
            eta_min=config.advantage_lr * config.min_lr_ratio
        )
    elif config.scheduler.lower() == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.advantage_lr * config.max_lr_multiplier,
            total_steps=config.max_steps,
            pct_start=config.warmup_steps / config.max_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations (slower but reproducible)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True  # Enable for better performance