"""
Graph Attention Network (GAT) Model

Uses attention mechanisms to weight neighbor contributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .base_gnn import BaseGNN


class GATModel(BaseGNN):
    """GAT model for graph representation learning."""

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 out_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.5,
                 attention_dropout: float = 0.2):
        """
        Initialize GAT model.

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (per head)
            out_dim: Output embedding dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate for features
            attention_dropout: Dropout rate for attention weights
        """
        super().__init__(hidden_dim, num_layers, dropout)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim * num_heads)

        # GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                # Intermediate layers: multi-head with concatenation
                self.convs.append(
                    GATConv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=num_heads,
                        concat=True,
                        dropout=attention_dropout
                    )
                )
            else:
                # Last layer: multi-head with averaging
                self.convs.append(
                    GATConv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=num_heads,
                        concat=False,
                        dropout=attention_dropout
                    )
                )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * num_heads if i < num_layers - 1 else hidden_dim)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_attention_weights: bool = False):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            return_attention_weights: Whether to return attention weights

        Returns:
            Node embeddings [num_nodes, out_dim]
            Or tuple of (embeddings, attention_weights) if return_attention_weights=True
        """
        # Input projection
        x = self.input_proj(x)
        x = F.elu(x)

        attention_weights_list = []

        # GAT layers
        for i, conv in enumerate(self.convs):
            if return_attention_weights:
                x_new, attn_weights = conv(x, edge_index, return_attention_weights=True)
                attention_weights_list.append(attn_weights)
            else:
                x_new = conv(x, edge_index)

            x_new = self.batch_norms[i](x_new)
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection (only if dimensions match)
            if i > 0 and x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new

        # Output projection
        x = self.output_proj(x)

        if return_attention_weights:
            return x, attention_weights_list
        else:
            return x
