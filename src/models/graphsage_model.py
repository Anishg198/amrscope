"""
GraphSAGE Model

Inductive graph learning using neighborhood sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .base_gnn import BaseGNN


class GraphSAGEModel(BaseGNN):
    """GraphSAGE model for graph representation learning."""

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 out_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 aggregator: str = 'mean'):
        """
        Initialize GraphSAGE model.

        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_layers: Number of SAGE layers
            dropout: Dropout rate
            aggregator: Aggregator type ('mean', 'max', 'lstm')
        """
        super().__init__(hidden_dim, num_layers, dropout)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # SAGE layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))

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

        # SAGE layers
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
