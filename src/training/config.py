"""
Configuration Management

Handles hyperparameters and training configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    """Configuration for model training and evaluation."""

    # Model architecture
    model_type: str = 'explainable_gat'  # 'gcn', 'sage', 'gat', 'hetero', 'explainable_gat'
    hidden_dim: int = 128
    out_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4  # For GAT models
    dropout: float = 0.5
    attention_dropout: float = 0.2

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    epochs: int = 200
    batch_size: int = 512  # For link prediction edge batches
    patience: int = 20  # Early stopping patience
    min_delta: float = 1e-4  # Minimum improvement for early stopping

    # Loss weights for multi-task learning
    link_prediction_weight: float = 1.0
    mechanism_classification_weight: float = 0.3
    regularization_weight: float = 0.01

    # Negative sampling
    num_neg_samples: int = 5  # Negative samples per positive edge
    neg_sampling_strategy: str = 'uniform'  # 'uniform' or 'hard'

    # Data
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # Evaluation
    eval_metrics: list = field(default_factory=lambda: ['mrr', 'hits@1', 'hits@3', 'hits@10', 'auc'])
    k_fold: int = 1  # Set to >1 for cross-validation

    # Device
    device: str = 'cuda'  # 'cuda' or 'cpu'
    mixed_precision: bool = True

    # Logging
    log_interval: int = 10  # Log every N epochs
    save_dir: str = 'results/models'
    exp_name: str = 'amr_gnn'
    use_wandb: bool = False
    wandb_project: str = 'amr-prediction'

    # Node/edge types (will be populated from data)
    node_types: list = field(default_factory=list)
    edge_types: list = field(default_factory=list)
    in_dims: dict = field(default_factory=dict)
    num_mechanisms: int = 5

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, \
            "Split ratios must sum to 1.0"
        assert self.hidden_dim > 0, "Hidden dimension must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.epochs > 0, "Number of epochs must be positive"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration
        """
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary of configuration parameters
        """
        return asdict(self)

    def update(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration parameter: {key}")


# Default configurations for different model types

def get_default_config(model_type: str = 'explainable_gat') -> Config:
    """
    Get default configuration for a specific model type.

    Args:
        model_type: Type of model

    Returns:
        Config object with default parameters
    """
    base_config = Config()

    if model_type == 'gcn':
        base_config.model_type = 'gcn'
        base_config.num_layers = 2
        base_config.hidden_dim = 128

    elif model_type == 'sage':
        base_config.model_type = 'sage'
        base_config.num_layers = 2
        base_config.hidden_dim = 128

    elif model_type == 'gat':
        base_config.model_type = 'gat'
        base_config.num_layers = 2
        base_config.num_heads = 8
        base_config.hidden_dim = 128

    elif model_type == 'hetero':
        base_config.model_type = 'hetero'
        base_config.num_layers = 3
        base_config.hidden_dim = 256

    elif model_type == 'explainable_gat':
        base_config.model_type = 'explainable_gat'
        base_config.num_layers = 3
        base_config.num_heads = 8
        base_config.hidden_dim = 256
        base_config.attention_dropout = 0.2

    return base_config
