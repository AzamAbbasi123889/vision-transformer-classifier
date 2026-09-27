"""
Vision Transformer Classifier - Main Entry Point

Demonstrates deep learning model training with PyTorch-style
implementation using only NumPy for portability.
"""

import numpy as np
from src.data_loader import generate_data
from src.model import NeuralNetwork
from src.trainer import Trainer
from src.config import EPOCHS, LEARNING_RATE, HIDDEN_SIZE, BATCH_SIZE


def main():
    """Run the deep learning training pipeline."""
    print("=" * 60)
    print("Vision Transformer Classifier")
    print("=" * 60)

    # Generate data
    print("\n[1/3] Generating dataset...")
    X_train, y_train, X_test, y_test = generate_data()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Create model
    print("\n[2/3] Building model...")
    input_dim = X_train.shape[1]
    model = NeuralNetwork(input_dim=input_dim, hidden_dim=HIDDEN_SIZE, output_dim=1)
    print(f"  Architecture: {input_dim} -> {HIDDEN_SIZE} -> 1")
    print(f"  Parameters: {model.count_parameters()}")

    # Train
    print("\n[3/3] Training...")
    trainer = Trainer(model, lr=LEARNING_RATE)
    history = trainer.train(X_train, y_train, X_test, y_test,
                           epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Final evaluation
    preds = model.predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Best Validation Loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
