"""
Generate baseline predictions for Case Study 2: Distribution Sampling

This script generates two samples per test point by:
1. Using the MLP to predict E[y|x]
2. Generating two samples by adding noise to represent uncertainty

This is a naive baseline. A better approach would model the bimodal distribution.

Outputs:
- case2/data/baseline_samples.npy: 100x2 matrix (two samples per test point)
"""

import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"
dataset_dir = script_dir.parent.parent / "dataset1" / "data"

# Create output directory if needed
data_dir.mkdir(parents=True, exist_ok=True)

# Load training data from dataset1
print("Loading training data from dataset1...")
train_data = np.load(dataset_dir / "train.npy")
train_x = train_data[:, 0].reshape(-1, 1)
train_y = train_data[:, 1]

print(f"Training data shape: {train_data.shape}")

# Load test data from dataset1
print("\nLoading test data from dataset1...")
test_x = np.load(dataset_dir / "test_x.npy").reshape(-1, 1)
test_y = np.load(dataset_dir / "test_y.npy")

print(f"Test X shape: {test_x.shape}, Test Y shape: {test_y.shape}")

# Train a tiny MLPRegressor
print("\nTraining tiny MLPRegressor (16 hidden units)...")
model = MLPRegressor(
    hidden_layer_sizes=(16,),
    max_iter=1000,
    random_state=42,
    early_stopping=False
)
model.fit(train_x, train_y)

print("Training complete!")

# Generate predictions (mean estimates)
print("\nGenerating baseline predictions...")
test_yhat = model.predict(test_x)

# Generate two samples per test point
# Naive strategy: add different noise to the mean prediction
# A better strategy would model the bimodal distribution properly
samples = np.zeros((test_x.shape[0], 2), dtype=np.float16)

for i in range(test_x.shape[0]):
    # Sample 1: mean + noise from N(0, 2)
    samples[i, 0] = test_yhat[i] + np.random.normal(0, 2)
    # Sample 2: mean + different noise from N(0, 2)
    samples[i, 1] = test_yhat[i] + np.random.normal(0, 2)

# Save samples
output_path = data_dir / "baseline_samples.npy"
np.save(output_path, samples)
print(f"Baseline samples saved to: {output_path}")

# Calculate energy score
print("\nCalculating energy score...")
sum_dist1 = np.sum(np.abs(test_y - samples[:, 0]))
sum_dist2 = np.sum(np.abs(test_y - samples[:, 1]))
sum_dist_samples = np.sum(np.abs(samples[:, 0] - samples[:, 1]))

energy_score = (sum_dist1 + sum_dist2) / (2 * len(test_y)) - 0.5 * sum_dist_samples / len(test_y)

print(f"{'='*60}")
print(f"Baseline Energy Score: {energy_score:.4f}")
print(f"{'='*60}")
print("This is a naive baseline that just adds noise to the mean prediction.")
print("A better approach would model the bimodal mixture distribution properly.")
