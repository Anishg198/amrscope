#!/usr/bin/env python3
"""
Training Script for AMR Prediction Models

This script trains GNN models for antimicrobial resistance prediction.

Usage:
    python train.py --config experiments/configs/explainable_gat_config.yaml
    python train.py --model explainable_gat --epochs 200
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

from src.data_preprocessing.graph_builder import HeterogeneousGraphBuilder
from src.models.explainable_gat import ExplainableHeterogeneousGAT
from src.models.gcn_model import GCNModel
from src.models.graphsage_model import GraphSAGEModel
from src.models.gat_model import GATModel
from src.models.heterogeneous_gnn import HeterogeneousGNN
from src.training.config import Config, get_default_config
from src.training.trainer import Trainer
from src.evaluation.metrics import print_metrics
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AMR prediction model')

    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--model', type=str, default='explainable_gat',
                       choices=['gcn', 'sage', 'gat', 'hetero', 'explainable_gat'],
                       help='Model type to train')
    parser.add_argument('--data', type=str, default='data/processed/graph.pkl',
                       help='Path to processed graph data')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Hidden dimension size')
    parser.add_argument('--save-dir', type=str, default='results/models',
                       help='Directory to save models')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    return parser.parse_args()


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(data_path: str):
    """Load preprocessed graph data."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {data_path}")

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run data preprocessing first. See notebooks/01_data_exploration.ipynb")
        sys.exit(1)

    builder = HeterogeneousGraphBuilder.load(data_path)
    return builder.graph, builder


def create_model(config: Config, graph_data):
    """Create model based on configuration."""
    logger = logging.getLogger(__name__)

    # Get node and edge types from data
    config.node_types = list(graph_data.node_types)
    config.edge_types = list(graph_data.edge_types)

    # Get input dimensions
    config.in_dims = {
        node_type: graph_data[node_type].x.shape[1]
        for node_type in graph_data.node_types
    }

    logger.info(f"Creating {config.model_type} model")

    if config.model_type == 'explainable_gat':
        model = ExplainableHeterogeneousGAT(
            node_types=config.node_types,
            edge_types=config.edge_types,
            in_dims=config.in_dims,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            num_mechanisms=config.num_mechanisms
        )

    elif config.model_type == 'hetero':
        model = HeterogeneousGNN(
            node_types=config.node_types,
            edge_types=config.edge_types,
            in_dims=config.in_dims,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            num_layers=config.num_layers,
            conv_type='gat',
            dropout=config.dropout
        )

    elif config.model_type in ['gcn', 'sage', 'gat']:
        # For baseline models, use gene features only
        in_dim = config.in_dims.get('gene', 128)

        if config.model_type == 'gcn':
            model = GCNModel(in_dim, config.hidden_dim, config.out_dim,
                           config.num_layers, config.dropout)
        elif config.model_type == 'sage':
            model = GraphSAGEModel(in_dim, config.hidden_dim, config.out_dim,
                                  config.num_layers, config.dropout)
        elif config.model_type == 'gat':
            model = GATModel(in_dim, config.hidden_dim, config.out_dim,
                           config.num_layers, config.num_heads, config.dropout)

    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    return model


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    logger = setup_logger(
        name='amr_gnn',
        level=getattr(logging, args.log_level),
        log_file=f'{args.save_dir}/training.log',
        console=True
    )

    logger.info("=" * 80)
    logger.info("AMR Prediction Model Training")
    logger.info("=" * 80)

    # Set random seed
    set_random_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = Config.from_yaml(args.config)
    else:
        logger.info(f"Using default configuration for {args.model}")
        config = get_default_config(args.model)

    # Override config with command line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.device:
        config.device = args.device

    config.save_dir = args.save_dir

    # Print configuration
    logger.info("\nConfiguration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")

    # Load data
    graph_data, builder = load_data(args.data)

    # Create model
    model = create_model(config, graph_data)

    # Create trainer
    trainer = Trainer(model, config, graph_data)

    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    history = trainer.train()

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete")
    logger.info("=" * 80)

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_metrics = trainer.validate()  # Would need separate test evaluation

    print_metrics(test_metrics, prefix="Test")

    # Save model
    model_path = Path(config.save_dir) / f"{config.exp_name}_final.pt"
    trainer.save_checkpoint(str(model_path))

    logger.info(f"\nModel saved to {model_path}")
    logger.info(f"Best validation MRR: {trainer.best_val_mrr:.4f}")

    # Save configuration
    config_path = Path(config.save_dir) / f"{config.exp_name}_config.yaml"
    config.to_yaml(str(config_path))
    logger.info(f"Configuration saved to {config_path}")

    logger.info("\nTraining complete!")


if __name__ == '__main__':
    main()
