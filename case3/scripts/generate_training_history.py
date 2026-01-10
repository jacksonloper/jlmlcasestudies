"""
Generate sample training history for Case Study 3: Modular Arithmetic

This generates synthetic training history data that demonstrates the 
"grokking" phenomenon - where training loss drops quickly (memorization)
but test loss remains high until suddenly dropping (generalization).

Outputs:
- case3/data/reference_training_loss.csv: Training and test loss over epochs
"""

import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

# Generate epochs - we'll show training over many epochs to demonstrate grokking
total_epochs = 10000
epochs = list(range(0, total_epochs + 1, 100))

# Training loss: drops quickly (memorization)
# Starts at ~4.57 (random guessing: -log(1/97)), drops rapidly to near 0
train_loss = []
for epoch in epochs:
    if epoch < 500:
        # Initial rapid drop
        loss = 4.57 * np.exp(-epoch / 100) + 0.01 * np.random.randn()
    else:
        # Near zero with small noise
        loss = 0.01 + 0.005 * np.random.randn()
    train_loss.append(max(0.001, loss))

# Test loss: stays high for a long time, then suddenly drops (grokking)
# This demonstrates delayed generalization
test_loss = []
grokking_epoch = 6000  # The epoch where grokking happens
for epoch in epochs:
    if epoch < grokking_epoch - 1000:
        # High loss during memorization phase
        # Slowly decreases but remains high
        base_loss = 4.57 - (epoch / grokking_epoch) * 1.5
        loss = base_loss + 0.1 * np.random.randn()
    elif epoch < grokking_epoch:
        # Start of transition
        progress = (epoch - (grokking_epoch - 1000)) / 1000
        base_loss = 3.0 - progress * 2.5
        loss = base_loss + 0.1 * np.random.randn()
    else:
        # After grokking - sudden drop
        progress = min(1.0, (epoch - grokking_epoch) / 500)
        loss = 0.5 * (1 - progress) + 0.01 + 0.005 * np.random.randn()
    test_loss.append(max(0.001, loss))

# Also generate accuracy data
train_accuracy = []
test_accuracy = []

for epoch in epochs:
    # Training accuracy: high quickly (memorization)
    if epoch < 500:
        acc = min(0.99, 0.1 + 0.9 * (1 - np.exp(-epoch / 100)))
    else:
        acc = 0.99 + 0.005 * np.random.randn()
    train_accuracy.append(min(1.0, max(0.0, acc)))
    
    # Test accuracy: low until grokking, then high
    if epoch < grokking_epoch - 1000:
        acc = 0.01 + epoch / grokking_epoch * 0.1
    elif epoch < grokking_epoch:
        progress = (epoch - (grokking_epoch - 1000)) / 1000
        acc = 0.15 + progress * 0.35
    else:
        progress = min(1.0, (epoch - grokking_epoch) / 500)
        acc = 0.5 + progress * 0.49
    test_accuracy.append(min(1.0, max(0.0, acc + 0.02 * np.random.randn())))

# Write to CSV
csv_path = output_dir / "reference_training_loss.csv"
with open(csv_path, 'w') as f:
    f.write("epoch,train_loss,test_loss,train_accuracy,test_accuracy\n")
    for i, epoch in enumerate(epochs):
        f.write(f"{epoch},{train_loss[i]:.6f},{test_loss[i]:.6f},{train_accuracy[i]:.6f},{test_accuracy[i]:.6f}\n")

print(f"Training history saved to {csv_path}")
print(f"Total epochs: {total_epochs}")
print(f"Grokking epoch: {grokking_epoch}")
print(f"Initial train loss: {train_loss[0]:.4f}")
print(f"Final train loss: {train_loss[-1]:.4f}")
print(f"Initial test loss: {test_loss[0]:.4f}")
print(f"Final test loss: {test_loss[-1]:.4f}")
