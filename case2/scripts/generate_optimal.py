"""
Generate optimal baseline predictions for Case Study 2: Distribution Sampling

This script generates two samples per test point by:
1. Sampling from the two modes of the true mixture distribution
2. One sample from N(10*cos(x), 1) and one from N(0, 1)

This is closer to optimal since it correctly models the bimodal structure.

Outputs:
- case2/data/optimal_samples.npy: 100x2 matrix (two samples per test point)
"""

import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"
dataset_dir = script_dir.parent.parent / "dataset1" / "data"

# Create output directory if needed
data_dir.mkdir(parents=True, exist_ok=True)

# Load test data from dataset1
print("Loading test data from dataset1...")
test_x = np.load(dataset_dir / "test_x.npy")
test_y = np.load(dataset_dir / "test_y.npy")

print(f"Test X shape: {test_x.shape}, Test Y shape: {test_y.shape}")

# Generate two samples per test point from the true mixture distribution
print("\nGenerating optimal samples from true mixture distribution...")
samples = np.zeros((test_x.shape[0], 2), dtype=np.float16)

for i in range(test_x.shape[0]):
    x = test_x[i]
    # Sample from the two modes of the mixture
    # Mode 1: N(10*cos(x), 1)
    samples[i, 0] = np.random.normal(10 * np.cos(x), 1)
    # Mode 2: N(0, 1)
    samples[i, 1] = np.random.normal(0, 1)

# Save samples
output_path = data_dir / "optimal_samples.npy"
np.save(output_path, samples)
print(f"Optimal samples saved to: {output_path}")

# Calculate energy score
print("\nCalculating energy score...")
sum_dist1 = np.sum(np.abs(test_y - samples[:, 0]))
sum_dist2 = np.sum(np.abs(test_y - samples[:, 1]))
sum_dist_samples = np.sum(np.abs(samples[:, 0] - samples[:, 1]))

energy_score = (sum_dist1 + sum_dist2) / (2 * len(test_y)) - 0.5 * sum_dist_samples / len(test_y)

print(f"{'='*60}")
print(f"Optimal Energy Score: {energy_score:.4f}")
print(f"{'='*60}")
print("This uses the true mixture distribution: one sample from each mode.")
print("This should be close to the best possible energy score.")

# Alternative: both samples from the full mixture
print("\nGenerating alternative samples (both from mixture)...")
samples_alt = np.zeros((test_x.shape[0], 2), dtype=np.float16)

for i in range(test_x.shape[0]):
    x = test_x[i]
    for j in range(2):
        # Sample from the full mixture
        if np.random.rand() < 0.5:
            samples_alt[i, j] = np.random.normal(10 * np.cos(x), 1)
        else:
            samples_alt[i, j] = np.random.normal(0, 1)

# Calculate energy score for alternative
sum_dist1_alt = np.sum(np.abs(test_y - samples_alt[:, 0]))
sum_dist2_alt = np.sum(np.abs(test_y - samples_alt[:, 1]))
sum_dist_samples_alt = np.sum(np.abs(samples_alt[:, 0] - samples_alt[:, 1]))

energy_score_alt = (sum_dist1_alt + sum_dist2_alt) / (2 * len(test_y)) - 0.5 * sum_dist_samples_alt / len(test_y)

print(f"Alternative (both from mixture) Energy Score: {energy_score_alt:.4f}")
print("\nBoth approaches should give similar (good) energy scores.")
