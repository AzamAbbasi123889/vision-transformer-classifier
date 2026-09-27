"""
Training loop implementation.
"""

import numpy as np
from typing import Dict, List


class Trainer:
    """Handles training loop with gradient descent."""

    def __init__(self, model, lr: float = 0.01):
        self.model = model
        self.lr = lr

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """Train the model and return loss history."""
        history = {"train_loss": [], "val_loss": []}
        n = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data
            idx = np.random.permutation(n)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward
                output = self.model.forward(X_batch)

                # Loss (binary cross-entropy)
                eps = 1e-8
                loss = -np.mean(
                    y_batch.reshape(-1, 1) * np.log(output + eps)
                    + (1 - y_batch.reshape(-1, 1)) * np.log(1 - output + eps)
                )
                epoch_loss += loss
                n_batches += 1

                # Backward
                grads = self.model.backward(y_batch)

                # Update weights
                self.model.W1 -= self.lr * grads["dW1"]
                self.model.b1 -= self.lr * grads["db1"]
                self.model.W2 -= self.lr * grads["dW2"]
                self.model.b2 -= self.lr * grads["db2"]

            # Validation loss
            val_out = self.model.forward(X_val)
            val_loss = -np.mean(
                y_val.reshape(-1, 1) * np.log(val_out + eps)
                + (1 - y_val.reshape(-1, 1)) * np.log(1 - val_out + eps)
            )

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 20 == 0 or epoch == 0:
                val_preds = self.model.predict(X_val)
                val_acc = np.mean(val_preds == y_val)
                print(
                    f"  Epoch {{epoch+1:>4}}/{epochs}"
                    f"  Loss: {{avg_train_loss:.4f}}"
                    f"  Val Loss: {{val_loss:.4f}}"
                    f"  Val Acc: {{val_acc:.4f}}"
                )

        return history
