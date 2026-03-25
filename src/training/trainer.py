"""
Model Trainer

Handles training loop, validation, and checkpointing for AMR prediction models.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import Config
from ..models.base_gnn import negative_sampling
from ..utils.early_stopping import EarlyStopping
from ..evaluation.metrics import compute_link_prediction_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for AMR GNN models."""

    def __init__(self,
                 model: nn.Module,
                 config: Config,
                 data: Optional[object] = None):
        """
        Initialize trainer.

        Args:
            model: GNN model to train
            config: Training configuration
            data: Graph data object (HeteroData)
        """
        self.model = model
        self.config = config
        self.data = data

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        if self.data is not None:
            self.data = self.data.to(self.device)

        # Setup optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='max'
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mrr': [],
            'val_hits@10': [],
        }

        # Best model state
        self.best_model_state = None
        self.best_val_mrr = 0.0

        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def compute_loss(self,
                    z_dict: Dict[str, torch.Tensor],
                    edge_index: torch.Tensor,
                    labels: torch.Tensor,
                    mechanism_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            z_dict: Node embeddings from model
            edge_index: Edge indices
            labels: Binary labels for link prediction
            mechanism_labels: Labels for mechanism classification (optional)

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # 1. Link prediction loss (binary cross-entropy)
        if hasattr(self.model, 'predict_links'):
            link_scores = self.model.predict_links(z_dict, edge_index, 'gene', 'drug')
            link_loss = F.binary_cross_entropy_with_logits(link_scores, labels.float())
            losses['link_prediction'] = link_loss * self.config.link_prediction_weight
        else:
            # For baseline models, use simpler approach
            gene_emb = z_dict.get('gene', z_dict)
            drug_emb = z_dict.get('drug', z_dict)

            src_emb = gene_emb[edge_index[0]]
            dst_emb = drug_emb[edge_index[1]]
            scores = (src_emb * dst_emb).sum(dim=-1)

            link_loss = F.binary_cross_entropy_with_logits(scores, labels.float())
            losses['link_prediction'] = link_loss * self.config.link_prediction_weight

        # 2. Mechanism classification loss (if applicable)
        if mechanism_labels is not None and hasattr(self.model, 'classify_mechanisms'):
            mechanism_logits = self.model.classify_mechanisms(z_dict, 'gene')
            mechanism_loss = F.cross_entropy(mechanism_logits, mechanism_labels)
            losses['mechanism_classification'] = mechanism_loss * self.config.mechanism_classification_weight

        # 3. Regularization loss (L2 on attention weights if available)
        if hasattr(self.model, 'attention_weights') and len(self.model.attention_weights) > 0:
            reg_loss = 0.0
            for key, (edge_idx, attn_weights) in self.model.attention_weights.items():
                reg_loss += (attn_weights ** 2).mean()
            losses['regularization'] = reg_loss * self.config.regularization_weight

        # Total loss
        losses['total'] = sum(losses.values())

        return losses

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Get training edges
        target_edge_type = ('gene', 'confers', 'drug')

        if target_edge_type not in self.data.edge_types:
            raise ValueError(f"Target edge type {target_edge_type} not found in data")

        edge_index = self.data[target_edge_type].edge_index
        train_mask = self.data[target_edge_type].train_mask

        train_edges = edge_index[:, train_mask]

        # Prepare node features
        x_dict = {
            node_type: self.data[node_type].x
            for node_type in self.data.node_types
        }

        # Prepare all edge indices (for message passing)
        edge_index_dict = {
            edge_type: self.data[edge_type].edge_index
            for edge_type in self.data.edge_types
        }

        # Forward pass to get embeddings
        z_dict = self.model(x_dict, edge_index_dict, return_attention=False)

        # Generate negative samples
        num_nodes = (self.data['gene'].num_nodes, self.data['drug'].num_nodes)
        neg_edges = negative_sampling(
            train_edges,
            num_nodes,
            num_neg_samples=self.config.num_neg_samples
        )

        # Combine positive and negative samples
        all_edges = torch.cat([train_edges, neg_edges], dim=1)
        labels = torch.cat([
            torch.ones(train_edges.shape[1]),
            torch.zeros(neg_edges.shape[1])
        ]).to(self.device)

        # Compute loss
        losses = self.compute_loss(z_dict, all_edges, labels)

        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return losses['total'].item()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        target_edge_type = ('gene', 'confers', 'drug')
        edge_index = self.data[target_edge_type].edge_index
        val_mask = self.data[target_edge_type].val_mask

        val_edges = edge_index[:, val_mask]

        # Prepare data
        x_dict = {
            node_type: self.data[node_type].x
            for node_type in self.data.node_types
        }

        edge_index_dict = {
            edge_type: self.data[edge_type].edge_index
            for edge_type in self.data.edge_types
        }

        # Forward pass
        z_dict = self.model(x_dict, edge_index_dict, return_attention=False)

        # Generate negative samples
        num_nodes = (self.data['gene'].num_nodes, self.data['drug'].num_nodes)
        neg_edges = negative_sampling(val_edges, num_nodes, num_neg_samples=10)

        # Combine edges
        all_edges = torch.cat([val_edges, neg_edges], dim=1)
        labels = torch.cat([
            torch.ones(val_edges.shape[1]),
            torch.zeros(neg_edges.shape[1])
        ]).to(self.device)

        # Compute metrics
        if hasattr(self.model, 'predict_links'):
            scores = self.model.predict_links(z_dict, all_edges, 'gene', 'drug')
        else:
            gene_emb = z_dict.get('gene', z_dict)
            drug_emb = z_dict.get('drug', z_dict)
            src_emb = gene_emb[all_edges[0]]
            dst_emb = drug_emb[all_edges[1]]
            scores = (src_emb * dst_emb).sum(dim=-1)

        metrics = compute_link_prediction_metrics(scores, labels)

        # Compute loss
        losses = self.compute_loss(z_dict, all_edges, labels)
        metrics['loss'] = losses['total'].item()

        return metrics

    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        Train model for specified number of epochs.

        Args:
            epochs: Number of epochs (uses config if not specified)

        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.epochs

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mrr'].append(val_metrics['mrr'])
            self.history['val_hits@10'].append(val_metrics['hits@10'])

            # Learning rate scheduling
            self.scheduler.step(val_metrics['mrr'])

            # Save best model
            if val_metrics['mrr'] > self.best_val_mrr:
                self.best_val_mrr = val_metrics['mrr']
                self.best_model_state = self.model.state_dict().copy()

            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                epoch_time = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val MRR: {val_metrics['mrr']:.4f} | "
                    f"Val Hits@10: {val_metrics['hits@10']:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )

            # Early stopping
            if self.early_stopping(val_metrics['mrr']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with Val MRR: {self.best_val_mrr:.4f}")

        return self.history

    def save_checkpoint(self, filepath: str):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
            'best_val_mrr': self.best_val_mrr,
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_mrr = checkpoint.get('best_val_mrr', 0.0)

        logger.info(f"Loaded checkpoint from {filepath}")
