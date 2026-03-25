"""
Early Stopping

Implements early stopping callback for training.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback to stop training when validation metric stops improving."""

    def __init__(self,
                 patience: int = 20,
                 min_delta: float = 1e-4,
                 mode: str = 'max'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize (e.g., accuracy), 'min' for metrics to minimize (e.g., loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_value = None
        self.should_stop = False

        if mode not in ['max', 'min']:
            raise ValueError("mode must be 'max' or 'min'")

    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            metric_value: Current metric value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False

        # Check if there's improvement
        if self.mode == 'max':
            improved = metric_value > self.best_value + self.min_delta
        else:
            improved = metric_value < self.best_value - self.min_delta

        if improved:
            self.best_value = metric_value
            self.counter = 0
            logger.debug(f"Metric improved to {metric_value:.6f}")
        else:
            self.counter += 1
            logger.debug(f"No improvement for {self.counter} epochs (best: {self.best_value:.6f})")

            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                self.should_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False
