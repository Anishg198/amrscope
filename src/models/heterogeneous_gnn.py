"""
Heterogeneous Graph Neural Network

Handles multiple node and edge types with type-specific transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv
from typing import Dict, Tuple, Optional


class HeterogeneousGNN(nn.Module):
    """Heterogeneous GNN with type-specific layers."""

    def __init__(self,
                 node_types: list,
                 edge_types: list,
                 in_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 out_dim: int = 128,
                 num_layers: int = 2,
                 conv_type: str = 'gcn',
                 dropout: float = 0.5):
        """
        Initialize heterogeneous GNN.

        Args:
            node_types: List of node type names
            edge_types: List of edge types as (src, rel, dst) tuples
            in_dims: Dictionary mapping node types to input dimensions
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_layers: Number of heterogeneous conv layers
            conv_type: Type of convolution ('gcn', 'sage', 'gat')
            dropout: Dropout rate
        """
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projections for each node type
        self.input_projs = nn.ModuleDict({
            node_type: nn.Linear(in_dims[node_type], hidden_dim)
            for node_type in node_types
        })

        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src_type, rel_type, dst_type = edge_type

                # Choose convolution type
                if conv_type == 'gcn':
                    conv_dict[edge_type] = GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
                elif conv_type == 'sage':
                    conv_dict[edge_type] = SAGEConv(hidden_dim, hidden_dim)
                elif conv_type == 'gat':
                    conv_dict[edge_type] = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, add_self_loops=False)

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        # Batch normalization for each node type
        self.batch_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.BatchNorm1d(hidden_dim)
                for node_type in node_types
            })
            for _ in range(num_layers)
        ])

        # Output projections for each node type
        self.output_projs = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, out_dim)
            for node_type in node_types
        })

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_dict: Dictionary mapping node types to feature tensors
            edge_index_dict: Dictionary mapping edge types to edge indices

        Returns:
            Dictionary mapping node types to embeddings
        """
        # Input projection
        x_dict = {
            node_type: F.relu(self.input_projs[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Heterogeneous convolutions
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)

            # Batch normalization and activation
            x_dict_new = {
                node_type: self.batch_norms[i][node_type](x)
                for node_type, x in x_dict_new.items()
            }

            x_dict_new = {
                node_type: F.relu(x)
                for node_type, x in x_dict_new.items()
            }

            x_dict_new = {
                node_type: F.dropout(x, p=self.dropout, training=self.training)
                for node_type, x in x_dict_new.items()
            }

            # Residual connection
            if i > 0:
                x_dict = {
                    node_type: x_dict[node_type] + x_dict_new[node_type]
                    for node_type in x_dict.keys()
                }
            else:
                x_dict = x_dict_new

        # Output projection
        x_dict = {
            node_type: self.output_projs[node_type](x)
            for node_type, x in x_dict.items()
        }

        return x_dict
