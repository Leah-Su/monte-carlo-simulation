import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def predictive_r2(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> float:
    """
    R^2 relative to the estimator based on the in-sample average.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - train_mean) ** 2)
    return float(1.0 - numerator / denominator)


