"""
Attention Visualization

Visualize attention weights for explainability.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
import torch


def visualize_attention_weights(model,
                                x_dict: Dict,
                                edge_index_dict: Dict,
                                gene_idx: int,
                                drug_idx: int,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize attention weights for a specific gene-drug prediction.

    Args:
        model: Trained model with attention mechanisms
        x_dict: Node features
        edge_index_dict: Edge indices
        gene_idx: Gene node index
        drug_idx: Drug node index
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Get explanation
    if not hasattr(model, 'explain_prediction'):
        raise ValueError("Model does not support explain_prediction method")

    explanation = model.explain_prediction(
        x_dict,
        edge_index_dict,
        gene_idx,
        drug_idx,
        top_k=10
    )

    # Extract attention paths
    attention_paths = explanation['attention_paths']

    if len(attention_paths) == 0:
        print("No attention paths found for this prediction")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)

    edge_types = [path['edge_type'] for path in attention_paths]
    attention_values = [path['attention'] for path in attention_paths]

    # Simplify edge type names for display
    edge_types_simplified = [et.split('_', 1)[-1] if '_' in et else et for et in edge_types]

    # Create horizontal bar plot
    y_pos = np.arange(len(edge_types_simplified))
    ax.barh(y_pos, attention_values, align='center', color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(edge_types_simplified)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Attention Weight', fontsize=12)
    ax.set_title(f'Top Attention Paths for Gene {gene_idx} → Drug {drug_idx}\n'
                f'Prediction Score: {explanation["score"]:.4f}',
                fontsize=14, fontweight='bold')

    # Add value labels on bars
    for i, v in enumerate(attention_values):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")

    plt.show()


def plot_attention_heatmap(attention_weights: torch.Tensor,
                           source_labels: Optional[list] = None,
                           target_labels: Optional[list] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8),
                           cmap: str = 'viridis'):
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention weight matrix [num_edges, num_heads] or [num_src, num_dst]
        source_labels: Labels for source nodes (optional)
        target_labels: Labels for target nodes (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size
        cmap: Colormap name
    """
    # Convert to numpy
    if torch.is_tensor(attention_weights):
        attn_np = attention_weights.cpu().numpy()
    else:
        attn_np = attention_weights

    # If multi-head, average across heads
    if attn_np.ndim > 2:
        attn_np = attn_np.mean(axis=-1)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        attn_np,
        cmap=cmap,
        center=0 if attn_np.min() < 0 else None,
        cbar_kws={'label': 'Attention Weight'},
        xticklabels=target_labels if target_labels else False,
        yticklabels=source_labels if source_labels else False,
        ax=ax
    )

    ax.set_title('Attention Weight Heatmap', fontsize=14, fontweight='bold')

    if source_labels:
        ax.set_ylabel('Source Nodes', fontsize=12)
    if target_labels:
        ax.set_xlabel('Target Nodes', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")

    plt.show()


def plot_attention_distribution(model,
                                layer_idx: int = 0,
                                edge_type: Optional[str] = None,
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 6)):
    """
    Plot distribution of attention weights for a specific layer/edge type.

    Args:
        model: Trained model with stored attention weights
        layer_idx: Layer index to visualize
        edge_type: Specific edge type to visualize (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if not hasattr(model, 'attention_weights') or not model.attention_weights:
        raise ValueError("Model has no stored attention weights. "
                        "Run forward() with return_attention=True first.")

    # Get attention weights
    attention_weights = model.get_attention_weights(edge_type=edge_type, layer=layer_idx)

    if len(attention_weights) == 0:
        print("No attention weights found for specified layer/edge type")
        return

    fig, axes = plt.subplots(len(attention_weights), 1,
                            figsize=(figsize[0], figsize[1] * len(attention_weights)))

    if len(attention_weights) == 1:
        axes = [axes]

    for ax, (key, (edge_idx, attn)) in zip(axes, attention_weights.items()):
        # Convert to numpy
        attn_np = attn.cpu().numpy() if torch.is_tensor(attn) else attn

        # Plot histogram
        ax.hist(attn_np.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Attention Weight', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Attention Distribution: {key}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention distribution to {save_path}")

    plt.show()
