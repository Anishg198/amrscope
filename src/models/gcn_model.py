"""
Graph Convolutional Network (GCN) Model

Baseline model using standard GCN layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .base_gnn import BaseGNN


class GCNModel(BaseGNN):
    """GCN model for graph representation learning."""

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 out_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        """
        Initialize GCN model.

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super().__init__(hidden_dim, num_layers, dropout)

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node embeddings [num_nodes, out_dim]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GCN layers
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new

        # Output projection
        x = self.output_proj(x)

        return x
