"""
Generate data for Case Study 1: Conditional Distribution Prediction

Generates:
- x ~ N(4, 1)
- y | x is an equal parts mixture of N(10*cos(x), 1) and N(0, 1)

Outputs:
- train.npy: 900x2 matrix (x, y pairs)
- test_x.npy: 100 vector (x values)
- test_y.npy: 100 vector (y values)
"""

import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 points
n_total = 1000
n_train = 900
n_test = 100

# Generate x values from N(4, 1)
x = np.random.normal(4, 1, n_total)

# Generate y values as mixture
y = np.zeros(n_total)
for i in range(n_total):
    # Equal parts mixture: flip a coin
    if np.random.rand() < 0.5:
        # From N(10*cos(x), 1)
        y[i] = np.random.normal(10 * np.cos(x[i]), 1)
    else:
        # From N(0, 1)
        y[i] = np.random.normal(0, 1)

# Split into train and test
train_x = x[:n_train]
train_y = y[:n_train]
test_x = x[n_train:]
test_y = y[n_train:]

# Create output directory (case1/scripts -> case1/data)
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

# Save as float16 to reduce file size
train = np.column_stack([train_x, train_y]).astype(np.float16)
test_x_arr = test_x.astype(np.float16)
test_y_arr = test_y.astype(np.float16)

np.save(output_dir / "train.npy", train)
np.save(output_dir / "test_x.npy", test_x_arr)
np.save(output_dir / "test_y.npy", test_y_arr)

print(f"Generated data saved to {output_dir}")
print(f"Train shape: {train.shape}")
print(f"Test X shape: {test_x_arr.shape}")
print(f"Test Y shape: {test_y_arr.shape}")
print(f"Train data statistics:")
print(f"  X: mean={train_x.mean():.2f}, std={train_x.std():.2f}")
print(f"  Y: mean={train_y.mean():.2f}, std={train_y.std():.2f}")
