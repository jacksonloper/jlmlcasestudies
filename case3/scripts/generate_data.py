"""
Generate data for Case Study 3: Modular Arithmetic Mod 97

This generates all 97^2 = 9409 possible (a, b) pairs where a, b ∈ {0, 1, ..., 96}
and computes (a + b) mod 97.

Input encoding: One-hot encoding of (a, b) as 97*2 = 194 binary features
where exactly two bits are on (one for a, one for b).

Output: Integer label in {0, 1, ..., 96} representing (a + b) mod 97.

Train/Test split: Half and half (approximately 4705/4704 split).

Outputs:
- train_x.npy: Training inputs as one-hot encoded vectors (shape: ~4705 x 194)
- train_y.npy: Training labels (shape: ~4705,)
- test_x.npy: Test inputs as one-hot encoded vectors (shape: ~4704 x 194)
- test_y.npy: Test labels (shape: ~4704,)
"""

import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Modular arithmetic parameters
MOD = 97
N_TOTAL = MOD * MOD  # 9409 total pairs

# Create all possible (a, b) pairs and their sums mod 97
pairs = []
labels = []
for a in range(MOD):
    for b in range(MOD):
        pairs.append((a, b))
        labels.append((a + b) % MOD)

pairs = np.array(pairs)
labels = np.array(labels)

# Create one-hot encoding for each pair
# First 97 bits for a, next 97 bits for b (194 total bits)
def one_hot_encode(pairs, mod=97):
    """Convert (a, b) pairs to one-hot encoding with 2*mod features."""
    n_samples = len(pairs)
    n_features = 2 * mod
    encoded = np.zeros((n_samples, n_features), dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        encoded[i, a] = 1.0  # First 97 bits: position a
        encoded[i, mod + b] = 1.0  # Next 97 bits: position b
    return encoded

# Encode all pairs
x_all = one_hot_encode(pairs, MOD)
y_all = labels.astype(np.int32)

# Half and half train/test split
indices = np.random.permutation(N_TOTAL)
n_train = N_TOTAL // 2 + (N_TOTAL % 2)  # 4705 if odd total
n_test = N_TOTAL - n_train  # 4704

train_indices = indices[:n_train]
test_indices = indices[n_train:]

train_x = x_all[train_indices]
train_y = y_all[train_indices]
test_x = x_all[test_indices]
test_y = y_all[test_indices]

# Create output directory
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

# Save data
np.save(output_dir / "train_x.npy", train_x)
np.save(output_dir / "train_y.npy", train_y)
np.save(output_dir / "test_x.npy", test_x)
np.save(output_dir / "test_y.npy", test_y)

print(f"Generated data saved to {output_dir}")
print(f"Train X shape: {train_x.shape}")
print(f"Train Y shape: {train_y.shape}")
print(f"Test X shape: {test_x.shape}")
print(f"Test Y shape: {test_y.shape}")
print(f"\nModular arithmetic: (a + b) mod {MOD}")
print(f"Input encoding: one-hot with {2 * MOD} features (exactly 2 bits on)")
print(f"Output: {MOD} classes (0 to {MOD - 1})")
print(f"\nTrain/Test split: {n_train}/{n_test} (half and half)")
