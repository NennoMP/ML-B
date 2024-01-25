"""
metrics.py

@description: Metrics for models evaluation.
"""

import numpy as np


def mean_euclidean_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Utility function to compute the Mean Euclidean Error (MEE) between 
    true and predicted values. Return the MEE score.

    Required arguments:
    - y_true: array containing true values (ground truth).
    - y_pred: array containing predicted values.
    """
    return np.mean(np.sqrt(np.sum(np.square(y_pred-y_true), axis=-1)))