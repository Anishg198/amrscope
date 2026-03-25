"""
Explainable Heterogeneous Graph Attention Network

Main novel contribution: combines heterogeneous GNN with explainable attention mechanisms.

Key features:
- Node-type-specific transformations
- Relation-type-specific attention
- Multi-task learning (link prediction + mechanism classification)
- Attention weight extraction for explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from typing import Dict, Tuple, Optional, List

from .base_gnn import LinkPredictor, MechanismClassifier


class RelationSpecificGATConv(nn.Module):
    """GAT convolution with relation-specific attention."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        """
        Initialize relation-specific GAT.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension (per head)
            num_heads: Number of attention heads
            dropout: Attention dropout rate
        """
        super().__init__()

        self.gat = GATConv(
            in_dim,
            out_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor],
                edge_index: torch.Tensor,
                return_attention_weights: bool = False):
        """
        Forward pass.

        Args:
            x: Tuple of (src_features, dst_features)
            edge_index: Edge indices
            return_attention_weights: Whether to return attention weights

        Returns:
            Output features and optionally attention weights
        """
        return self.gat(x, edge_index, return_attention_weights=return_attention_weights)


class ExplainableHeterogeneousGAT(nn.Module):
    """
    Explainable Heterogeneous Graph Attention Network.

    This is the main novel model that combines:
    1. Heterogeneous graph structure
    2. Relation-specific attention mechanisms
    3. Multi-task learning
    4. Explainability through attention weights
    """

    def __init__(self,
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 in_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 out_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.5,
                 attention_dropout: float = 0.2,
                 num_mechanisms: int = 5):
        """
        Initialize explainable heterogeneous GAT.

        Args:
            node_types: List of node type names
            edge_types: List of edge types as (src, rel, dst) tuples
            in_dims: Dictionary mapping node types to input dimensions
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_layers: Number of heterogeneous GAT layers
            num_heads: Number of attention heads
            dropout: Feature dropout rate
            attention_dropout: Attention dropout rate
            num_mechanisms: Number of resistance mechanism classes
        """
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.in_dims = in_dims
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_mechanisms = num_mechanisms

        # Store attention weights for explainability
        self.attention_weights = {}

        # Input projections for each node type
        self.input_projs = nn.ModuleDict({
            node_type: Linear(in_dims.get(node_type, hidden_dim), hidden_dim)
            for node_type in node_types
        })

        # Heterogeneous GAT layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}

            for edge_type in edge_types:
                src_type, rel_type, dst_type = edge_type

                # Relation-specific GAT
                conv_dict[edge_type] = RelationSpecificGATConv(
                    hidden_dim,
                    hidden_dim,
                    num_heads=num_heads,
                    dropout=attention_dropout
                )

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.BatchNorm1d(hidden_dim)
                for node_type in node_types
            })
            for _ in range(num_layers)
        ])

        # Output projections
        self.output_projs = nn.ModuleDict({
            node_type: Linear(hidden_dim, out_dim)
            for node_type in node_types
        })

        # Task-specific heads

        # 1. Link prediction head (gene -> drug)
        self.link_predictor = LinkPredictor(out_dim, hidden_dim=64)

        # 2. Mechanism classification head (for genes)
        if num_mechanisms > 0:
            self.mechanism_classifier = MechanismClassifier(
                out_dim,
                num_mechanisms,
                hidden_dim=64
            )

    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass to get node embeddings.

        Args:
            x_dict: Dictionary mapping node types to feature tensors
            edge_index_dict: Dictionary mapping edge types to edge indices
            return_attention: Whether to store attention weights

        Returns:
            Dictionary mapping node types to embeddings
        """
        # Clear previous attention weights
        if return_attention:
            self.attention_weights = {}

        # Input projection
        x_dict = {
            node_type: F.elu(self.input_projs[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Heterogeneous GAT layers
        for layer_idx, conv in enumerate(self.convs):
            # Use HeteroConv directly - it handles all edge types
            x_dict_new = conv(x_dict, edge_index_dict)

            # Ensure x_dict_new only contains tensors
            x_dict_new = {k: v for k, v in x_dict_new.items() if isinstance(v, torch.Tensor)}

            # Batch normalization and activation
            for node_type in x_dict_new.keys():
                x_dict_new[node_type] = self.batch_norms[layer_idx][node_type](x_dict_new[node_type])
                x_dict_new[node_type] = F.elu(x_dict_new[node_type])
                x_dict_new[node_type] = F.dropout(
                    x_dict_new[node_type],
                    p=self.dropout,
                    training=self.training
                )

            # Residual connection
            if layer_idx > 0:
                for node_type in x_dict_new.keys():
                    if node_type in x_dict:
                        x_dict[node_type] = x_dict[node_type] + x_dict_new[node_type]
            else:
                x_dict = x_dict_new

        # Output projection
        x_dict = {
            node_type: self.output_projs[node_type](x)
            for node_type, x in x_dict.items()
        }

        return x_dict

    def predict_links(self,
                     z_dict: Dict[str, torch.Tensor],
                     edge_index: torch.Tensor,
                     src_type: str = 'gene',
                     dst_type: str = 'drug') -> torch.Tensor:
        """
        Predict link scores for gene-drug pairs.

        Args:
            z_dict: Node embeddings from forward pass
            edge_index: Edge indices to predict [2, num_edges]
            src_type: Source node type
            dst_type: Destination node type

        Returns:
            Link scores [num_edges]
        """
        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]

        scores = self.link_predictor(z_src, z_dst, edge_index)

        return scores

    def predict_all_links(self,
                         z_dict: Dict[str, torch.Tensor],
                         src_type: str = 'gene',
                         dst_type: str = 'drug') -> torch.Tensor:
        """
        Predict scores for all possible gene-drug links.

        Args:
            z_dict: Node embeddings from forward pass
            src_type: Source node type
            dst_type: Destination node type

        Returns:
            Link scores [num_src, num_dst]
        """
        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]

        scores = self.link_predictor.predict_all(z_src, z_dst)

        return scores

    def classify_mechanisms(self,
                           z_dict: Dict[str, torch.Tensor],
                           node_type: str = 'gene') -> torch.Tensor:
        """
        Classify resistance mechanisms for genes.

        Args:
            z_dict: Node embeddings from forward pass
            node_type: Node type to classify

        Returns:
            Class logits [num_nodes, num_classes]
        """
        z = z_dict[node_type]
        logits = self.mechanism_classifier(z)

        return logits

    def get_attention_weights(self,
                             edge_type: Optional[str] = None,
                             layer: Optional[int] = None) -> Dict:
        """
        Get stored attention weights.

        Args:
            edge_type: Specific edge type to retrieve (None for all)
            layer: Specific layer to retrieve (None for all)

        Returns:
            Dictionary of attention weights
        """
        if not self.attention_weights:
            raise ValueError("No attention weights stored. Run forward() with return_attention=True first.")

        if edge_type is None and layer is None:
            return self.attention_weights

        filtered = {}
        for key, value in self.attention_weights.items():
            if edge_type and edge_type not in key:
                continue
            if layer is not None and f"layer{layer}" not in key:
                continue
            filtered[key] = value

        return filtered

    def explain_prediction(self,
                          x_dict: Dict[str, torch.Tensor],
                          edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                          gene_idx: int,
                          drug_idx: int,
                          top_k: int = 10) -> Dict:
        """
        Explain a specific gene-drug prediction.

        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            gene_idx: Gene node index
            drug_idx: Drug node index
            top_k: Number of top attention paths to return

        Returns:
            Dictionary with explanation including attention paths
        """
        # Forward pass with attention
        z_dict = self.forward(x_dict, edge_index_dict, return_attention=True)

        # Get prediction score
        edge_to_explain = torch.tensor([[gene_idx], [drug_idx]], dtype=torch.long)
        score = self.predict_links(z_dict, edge_to_explain, 'gene', 'drug')

        # Extract relevant attention weights
        explanation = {
            'gene_idx': gene_idx,
            'drug_idx': drug_idx,
            'score': score.item(),
            'attention_paths': []
        }

        # Get attention weights for paths involving this gene and drug
        for key, (edge_idx, attn_weights) in self.attention_weights.items():
            # Find edges involving the gene or drug
            src_mask = edge_idx[0] == gene_idx
            dst_mask = edge_idx[1] == drug_idx

            if src_mask.any() or dst_mask.any():
                relevant_attn = attn_weights[src_mask | dst_mask]
                if len(relevant_attn) > 0:
                    explanation['attention_paths'].append({
                        'edge_type': key,
                        'attention': relevant_attn.mean().item()
                    })

        # Sort by attention weight
        explanation['attention_paths'] = sorted(
            explanation['attention_paths'],
            key=lambda x: x['attention'],
            reverse=True
        )[:top_k]

        return explanation
