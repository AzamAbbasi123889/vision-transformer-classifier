"""
Neural network implementation from scratch using NumPy.
"""

import numpy as np


class NeuralNetwork:
    """Two-layer neural network with ReLU activation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * scale2
        self.b2 = np.zeros(output_dim)

        # Cache for backprop
        self._cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        a2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -500, 500)))  # Sigmoid

        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2

    def backward(self, y: np.ndarray) -> dict:
        """Backward pass returning gradients."""
        m = y.shape[0]
        c = self._cache
        a2 = c["a2"]

        # Output layer
        dz2 = a2 - y.reshape(-1, 1)
        dW2 = (c["a1"].T @ dz2) / m
        db2 = np.mean(dz2, axis=0)

        # Hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (c["z1"] > 0).astype(float)  # ReLU gradient
        dW1 = (c["X"].T @ dz1) / m
        db1 = np.mean(dz1, axis=0)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        probs = self.forward(X)
        return (probs >= 0.5).astype(int).flatten()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size
