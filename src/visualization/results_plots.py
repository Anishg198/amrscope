"""
Results Plotting

Visualize training results and model comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd


def plot_training_curves(history: Dict,
                         metrics: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)):
    """
    Plot training curves (loss, metrics over epochs).

    Args:
        history: Training history dictionary
        metrics: List of metrics to plot (default: all available)
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if metrics is None:
        # Plot all metrics except loss
        metrics = [k for k in history.keys() if 'loss' not in k]

    # Create subplots
    num_plots = 1 + len(metrics)  # 1 for loss, rest for metrics
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)

    if num_plots == 1:
        axes = [axes]

    # Plot loss
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot metrics
    for idx, metric_name in enumerate(metrics, 1):
        if metric_name not in history:
            continue

        ax = axes[idx]
        ax.plot(history[metric_name], linewidth=2, color='green')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_name.upper(), fontsize=11)
        ax.set_title(f'Validation {metric_name.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()


def plot_metrics_comparison(results: Dict[str, Dict],
                           metrics: Optional[List[str]] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6)):
    """
    Plot comparison of metrics across different models.

    Args:
        results: Dictionary mapping model names to their metrics
        metrics: List of metrics to compare (default: all available)
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if metrics is None:
        # Get all unique metrics
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        metrics = sorted(list(all_metrics))

    # Filter out non-numeric metrics
    numeric_metrics = []
    for metric in metrics:
        for model_results in results.values():
            if metric in model_results and isinstance(model_results[metric], (int, float)):
                numeric_metrics.append(metric)
                break

    if not numeric_metrics:
        print("No numeric metrics found for comparison")
        return

    # Prepare data
    model_names = list(results.keys())
    num_metrics = len(numeric_metrics)

    # Create subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(numeric_metrics):
        ax = axes[idx]

        # Get values for this metric
        values = [results[model].get(metric, 0) for model in model_names]

        # Create bar plot
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, values, align='center', alpha=0.8)

        # Color bars
        colors = sns.color_palette("husl", len(model_names))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")

    plt.show()


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 8),
                         normalize: bool = False):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix [num_classes, num_classes]
        class_names: List of class names (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        xticklabels=class_names if class_names else range(cm.shape[0]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_roc_curve(fpr: np.ndarray,
                  tpr: np.ndarray,
                  auc: float,
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (8, 8)):
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: AUC value
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")

    plt.show()
