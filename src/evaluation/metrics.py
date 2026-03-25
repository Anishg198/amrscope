"""
Evaluation Metrics

Implements standard metrics for link prediction and classification.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    average_precision_score
)
from typing import Dict, Tuple, Optional


def compute_mrr(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank.

    For each query (positive sample), rank all candidates and compute
    reciprocal rank of the true positive.

    Args:
        scores: Predicted scores [num_samples]
        labels: Binary labels (1 for positive, 0 for negative) [num_samples]

    Returns:
        MRR value
    """
    if scores.shape != labels.shape:
        raise ValueError("Scores and labels must have same shape")

    # Convert to numpy
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores_np)
    sorted_labels = labels_np[sorted_indices]

    # Find ranks of positive samples (1-indexed)
    positive_indices = np.where(sorted_labels == 1)[0]

    if len(positive_indices) == 0:
        return 0.0

    ranks = positive_indices + 1

    # Compute reciprocal ranks
    reciprocal_ranks = 1.0 / ranks

    # Mean reciprocal rank
    mrr = np.mean(reciprocal_ranks)

    return float(mrr)


def compute_hits_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """
    Compute Hits@k metric.

    Proportion of positive samples that appear in top-k predictions.

    Args:
        scores: Predicted scores [num_samples]
        labels: Binary labels (1 for positive, 0 for negative) [num_samples]
        k: Cutoff for hits

    Returns:
        Hits@k value
    """
    # Convert to numpy
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores_np)
    sorted_labels = labels_np[sorted_indices]

    # Check if any positive sample is in top-k
    top_k_labels = sorted_labels[:k]
    num_positives_in_topk = np.sum(top_k_labels)

    # Total number of positives
    num_positives = np.sum(labels_np)

    if num_positives == 0:
        return 0.0

    hits = num_positives_in_topk / num_positives

    return float(hits)


def compute_mean_rank(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Mean Rank.

    Average rank of positive samples.

    Args:
        scores: Predicted scores [num_samples]
        labels: Binary labels (1 for positive, 0 for negative) [num_samples]

    Returns:
        Mean rank value
    """
    # Convert to numpy
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores_np)
    sorted_labels = labels_np[sorted_indices]

    # Find ranks of positive samples (1-indexed)
    positive_indices = np.where(sorted_labels == 1)[0]

    if len(positive_indices) == 0:
        return float('inf')

    ranks = positive_indices + 1

    # Mean rank
    mr = np.mean(ranks)

    return float(mr)


def compute_auc_roc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute AUC-ROC.

    Args:
        scores: Predicted scores [num_samples]
        labels: Binary labels (1 for positive, 0 for negative) [num_samples]

    Returns:
        AUC-ROC value
    """
    # Convert to numpy
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Check if we have both classes
    if len(np.unique(labels_np)) < 2:
        return 0.5

    try:
        auc = roc_auc_score(labels_np, scores_np)
    except ValueError:
        auc = 0.5

    return float(auc)


def compute_link_prediction_metrics(scores: torch.Tensor,
                                    labels: torch.Tensor,
                                    k_values: Optional[list] = None) -> Dict[str, float]:
    """
    Compute all link prediction metrics.

    Args:
        scores: Predicted scores [num_samples]
        labels: Binary labels (1 for positive, 0 for negative) [num_samples]
        k_values: List of k values for Hits@k (default: [1, 3, 10])

    Returns:
        Dictionary of metrics
    """
    if k_values is None:
        k_values = [1, 3, 10]

    metrics = {}

    # MRR
    metrics['mrr'] = compute_mrr(scores, labels)

    # Mean Rank
    metrics['mean_rank'] = compute_mean_rank(scores, labels)

    # Hits@k
    for k in k_values:
        metrics[f'hits@{k}'] = compute_hits_at_k(scores, labels, k=k)

    # AUC-ROC
    metrics['auc'] = compute_auc_roc(scores, labels)

    # Average Precision
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    if len(np.unique(labels_np)) >= 2:
        try:
            metrics['avg_precision'] = float(average_precision_score(labels_np, scores_np))
        except:
            metrics['avg_precision'] = 0.0
    else:
        metrics['avg_precision'] = 0.0

    return metrics


def compute_classification_metrics(predictions: torch.Tensor,
                                   labels: torch.Tensor,
                                   num_classes: Optional[int] = None) -> Dict:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted class labels [num_samples] or logits [num_samples, num_classes]
        labels: True class labels [num_samples]
        num_classes: Number of classes (inferred if not provided)

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if predictions.ndim > 1:
        # Logits to class predictions
        predictions_np = predictions.argmax(dim=-1).cpu().numpy()
    else:
        predictions_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions

    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np,
        predictions_np,
        average='weighted',
        zero_division=0
    )

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_np,
        predictions_np,
        average='macro',
        zero_division=0
    )

    # Accuracy
    accuracy = (predictions_np == labels_np).mean()

    # Confusion matrix
    cm = confusion_matrix(labels_np, predictions_np)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'confusion_matrix': cm,
    }

    return metrics


def compute_ranking_metrics_per_query(scores_matrix: np.ndarray,
                                      true_indices: np.ndarray) -> Dict[str, float]:
    """
    Compute ranking metrics for each query separately.

    Useful for link prediction where each source node is a query.

    Args:
        scores_matrix: Score matrix [num_queries, num_candidates]
        true_indices: True target indices for each query [num_queries]

    Returns:
        Dictionary of averaged metrics
    """
    num_queries = scores_matrix.shape[0]

    mrr_list = []
    hits_1_list = []
    hits_3_list = []
    hits_10_list = []
    mean_rank_list = []

    for i in range(num_queries):
        scores = scores_matrix[i]
        true_idx = true_indices[i]

        # Rank candidates
        ranked_indices = np.argsort(-scores)
        rank = np.where(ranked_indices == true_idx)[0][0] + 1  # 1-indexed

        # MRR
        mrr_list.append(1.0 / rank)

        # Hits@k
        hits_1_list.append(1.0 if rank <= 1 else 0.0)
        hits_3_list.append(1.0 if rank <= 3 else 0.0)
        hits_10_list.append(1.0 if rank <= 10 else 0.0)

        # Mean rank
        mean_rank_list.append(rank)

    metrics = {
        'mrr': np.mean(mrr_list),
        'hits@1': np.mean(hits_1_list),
        'hits@3': np.mean(hits_3_list),
        'hits@10': np.mean(hits_10_list),
        'mean_rank': np.mean(mean_rank_list),
    }

    return metrics


def print_metrics(metrics: Dict, prefix: str = ""):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for printing (e.g., "Test")
    """
    print(f"\n{prefix} Metrics:" if prefix else "\nMetrics:")
    print("-" * 50)

    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"{key}:")
            print(value)
        elif isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    print("-" * 50)
