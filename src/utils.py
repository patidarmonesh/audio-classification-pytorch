"""Utility functions for training and evaluation."""

import random
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(y_true, y_pred, average='macro'):
    """Calculate classification metrics."""
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    return f1, precision, recall


def compute_class_weights(labels, num_classes):
    """Compute class weights for imbalanced datasets."""
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights
