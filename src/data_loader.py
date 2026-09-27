"""
Data generation for deep learning demonstration.
"""

import numpy as np
from typing import Tuple


def generate_data(
    n_train: int = 800,
    n_test: int = 200,
    n_features: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a non-linear classification dataset."""
    rng = np.random.RandomState(seed)

    def _make_data(n):
        X = rng.randn(n, n_features)
        # Non-linear decision boundary
        y = (
            np.sin(X[:, 0] * 2) + X[:, 1] ** 2 - X[:, 2] * X[:, 3]
            + 0.5 * np.cos(X[:, 4]) + rng.randn(n) * 0.3
        )
        y = (y > np.median(y)).astype(np.float64)
        return X, y

    X_train, y_train = _make_data(n_train)
    X_test, y_test = _make_data(n_test)
    return X_train, y_train, X_test, y_test
