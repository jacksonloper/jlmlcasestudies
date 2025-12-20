"""
Train a HistGradientBoostingRegressor baseline for Case Study 1

This script:
1. Loads the training data
2. Trains a HistGradientBoostingRegressor with default parameters
3. Generates predictions on the test set
4. Saves predictions as hgb_test_yhat.npy
5. Calculates and reports MSE scores:
   - Baseline MSE (HGB predictions)
   - Best possible MSE (using ground truth: E[y|x] = 0.5 * x^2)
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"

# Load training data
print("Loading training data...")
train_data = np.load(data_dir / "train.npy")
train_x = train_data[:, 0].reshape(-1, 1)  # Reshape for sklearn
train_y = train_data[:, 1]

print(f"Training data shape: {train_data.shape}")
print(f"Train X shape: {train_x.shape}, Train Y shape: {train_y.shape}")

# Load test data
print("\nLoading test data...")
test_x = np.load(data_dir / "test_x.npy").reshape(-1, 1)  # Reshape for sklearn
test_y = np.load(data_dir / "test_y.npy")

print(f"Test X shape: {test_x.shape}, Test Y shape: {test_y.shape}")

# Train HistGradientBoostingRegressor with default parameters
print("\nTraining HistGradientBoostingRegressor with default parameters...")
model = HistGradientBoostingRegressor(random_state=42)
model.fit(train_x, train_y)

print("Training complete!")

# Generate predictions on test set
print("\nGenerating predictions on test set...")
test_yhat = model.predict(test_x)

# Save predictions
output_path = data_dir / "hgb_test_yhat.npy"
np.save(output_path, test_yhat.astype(np.float16))
print(f"Predictions saved to: {output_path}")

# Calculate MSE for HGB predictions
mse_hgb = np.mean((test_yhat - test_y) ** 2)
print(f"\n{'='*60}")
print(f"HGB Baseline MSE: {mse_hgb:.4f}")
print(f"{'='*60}")
print("This is a not-awful baseline score using default HGB parameters.")

# Calculate best possible MSE using ground truth expected value
# The true distribution is: y | x ~ 0.5 * N(x^2, 1) + 0.5 * N(0, 1)
# The optimal prediction minimizing MSE is the expected value:
# E[y | x] = 0.5 * x^2 + 0.5 * 0 = 0.5 * x^2
print(f"\n{'='*60}")
print("Calculating best possible MSE using ground truth...")
test_x_flat = test_x.flatten()
optimal_predictions = 0.5 * test_x_flat ** 2
mse_optimal = np.mean((optimal_predictions - test_y) ** 2)
print(f"Best Possible MSE (using E[y|x] = 0.5 * x²): {mse_optimal:.4f}")
print(f"{'='*60}")
print("This is the essentially best-possible score (expected MSE).")

# Print summary
print(f"\n{'='*60}")
print("SUMMARY:")
print(f"  HGB Baseline MSE:    {mse_hgb:.4f} (not-awful)")
print(f"  Best Possible MSE:   {mse_optimal:.4f} (optimal)")
print(f"  Difference:          {mse_hgb - mse_optimal:.4f}")
print(f"{'='*60}")
