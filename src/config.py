"""Configuration for the deep learning project."""

import os

EPOCHS = int(os.getenv("EPOCHS", "100"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "32"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
