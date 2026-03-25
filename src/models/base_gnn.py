"""
Base GNN Module

Provides base class and common functionality for all GNN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod


class BaseGNN(nn.Module, ABC):
    """Base class for all GNN models."""

    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        """
        Initialize base GNN.

        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    @abstractmethod
    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Node embeddings
        """
        pass

    def reset_parameters(self):
        """Reset all model parameters."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def get_embeddings(self, x, edge_index):
        """
        Get node embeddings (same as forward, but clearer naming).

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Node embeddings
        """
        return self.forward(x, edge_index)


class LinkPredictor(nn.Module):
    """Link prediction head using dot product."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        """
        Initialize link predictor.

        Args:
            in_dim: Input embedding dimension
            hidden_dim: Hidden dimension for MLP (optional)
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Optional MLP before dot product
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict link scores.

        Args:
            z_src: Source node embeddings [num_src, dim]
            z_dst: Destination node embeddings [num_dst, dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Link scores [num_edges]
        """
        # Optional transformation
        z_src = self.mlp(z_src)
        z_dst = self.mlp(z_dst)

        # Get embeddings for edges
        src_emb = z_src[edge_index[0]]
        dst_emb = z_dst[edge_index[1]]

        # Dot product
        scores = (src_emb * dst_emb).sum(dim=-1)

        return scores

    def predict_all(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for all possible links.

        Args:
            z_src: Source node embeddings [num_src, dim]
            z_dst: Destination node embeddings [num_dst, dim]

        Returns:
            Scores for all links [num_src, num_dst]
        """
        z_src = self.mlp(z_src)
        z_dst = self.mlp(z_dst)

        # Matrix multiplication: [num_src, dim] @ [dim, num_dst]
        scores = z_src @ z_dst.t()

        return scores


class MechanismClassifier(nn.Module):
    """Multi-class classifier for resistance mechanisms."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 64):
        """
        Initialize mechanism classifier.

        Args:
            in_dim: Input embedding dimension
            num_classes: Number of mechanism classes
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify mechanism.

        Args:
            x: Node embeddings [num_nodes, dim]

        Returns:
            Class logits [num_nodes, num_classes]
        """
        return self.mlp(x)


def negative_sampling(edge_index: torch.Tensor,
                     num_nodes: Tuple[int, int],
                     num_neg_samples: int = 1) -> torch.Tensor:
    """
    Generate negative samples for link prediction.

    Args:
        edge_index: Positive edge indices [2, num_edges]
        num_nodes: Tuple of (num_src_nodes, num_dst_nodes)
        num_neg_samples: Number of negative samples per positive edge

    Returns:
        Negative edge indices [2, num_neg_edges]
    """
    num_src, num_dst = num_nodes
    num_pos_edges = edge_index.shape[1]

    # Create set of positive edges for fast lookup
    pos_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Generate negative samples
    neg_edges = []
    num_neg_needed = num_pos_edges * num_neg_samples

    while len(neg_edges) < num_neg_needed:
        # Random sample
        src = torch.randint(0, num_src, (num_neg_needed - len(neg_edges),))
        dst = torch.randint(0, num_dst, (num_neg_needed - len(neg_edges),))

        # Filter out positive edges
        for s, d in zip(src.tolist(), dst.tolist()):
            if (s, d) not in pos_edges:
                neg_edges.append((s, d))
            if len(neg_edges) >= num_neg_needed:
                break

    neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t()

    return neg_edge_index


def compute_mrr(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        scores: Predicted scores [num_samples]
        labels: Binary labels (1 for positive, 0 for negative) [num_samples]

    Returns:
        MRR value
    """
    # Sort scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]

    # Find ranks of positive samples (1-indexed)
    ranks = torch.where(sorted_labels == 1)[0] + 1

    # Compute reciprocal ranks
    reciprocal_ranks = 1.0 / ranks.float()

    # Mean reciprocal rank
    mrr = reciprocal_ranks.mean().item()

    return mrr


def compute_hits_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """
    Compute Hits@k metric.

    Args:
        scores: Predicted scores [num_samples]
        labels: Binary labels (1 for positive, 0 for negative) [num_samples]
        k: Cutoff for hits

    Returns:
        Hits@k value
    """
    # Sort scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]

    # Check if any positive sample is in top-k
    top_k_labels = sorted_labels[:k]
    hits = (top_k_labels.sum() > 0).float().item()

    return hits
