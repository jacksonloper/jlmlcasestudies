"""
Generate ground truth optimal samples for Case Study 2: Distribution Sampling

This script generates samples by directly sampling from the true mixture distribution.
This represents the best possible performance (oracle access to true distribution).

Outputs:
- case2/data/groundtruth_samples.npy: 100x2 matrix (two samples per test point)
"""

import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"
# Dataset is at root level
dataset_dir = script_dir.parent.parent.parent / "dataset1" / "data"

# Create output directory if needed
data_dir.mkdir(parents=True, exist_ok=True)

# Load test data from dataset1
print("Loading test data from dataset1...")
test_x = np.load(dataset_dir / "test_x.npy")
test_y = np.load(dataset_dir / "test_y.npy")

print(f"Test X shape: {test_x.shape}, Test Y shape: {test_y.shape}")

# Generate two samples per test point from the true mixture distribution
print("\nGenerating ground truth samples from true mixture distribution...")
samples = np.zeros((test_x.shape[0], 2), dtype=np.float16)

for i in range(test_x.shape[0]):
    x = test_x[i]
    # Sample from the two modes of the mixture
    # Mode 1: N(10*cos(x), 1)
    samples[i, 0] = np.random.normal(10 * np.cos(x), 1)
    # Mode 2: N(0, 1)
    samples[i, 1] = np.random.normal(0, 1)

# Save samples
output_path = data_dir / "groundtruth_samples.npy"
np.save(output_path, samples)
print(f"Ground truth samples saved to: {output_path}")

# Calculate energy score
print("\nCalculating energy score...")
sum_dist1 = np.sum(np.abs(test_y - samples[:, 0]))
sum_dist2 = np.sum(np.abs(test_y - samples[:, 1]))
sum_dist_samples = np.sum(np.abs(samples[:, 0] - samples[:, 1]))

energy_score = (sum_dist1 + sum_dist2) / (2 * len(test_y)) - 0.5 * sum_dist_samples / len(test_y)

print(f"{'='*60}")
print(f"Ground Truth Energy Score: {energy_score:.4f}")
print(f"{'='*60}")
print("This uses the true mixture distribution: one sample from each mode.")
print("This represents the best possible energy score (oracle performance).")
