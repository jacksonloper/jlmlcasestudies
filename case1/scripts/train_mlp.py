"""
Train a tiny MLP (Multi-Layer Perceptron) baseline for Case Study 1

This script:
1. Loads the training data
2. Trains a small MLPRegressor with default parameters
3. Generates predictions on the test set
4. Saves predictions as mlp_test_yhat.npy
5. Calculates and reports RMSE scores:
   - Baseline RMSE (MLP predictions)
   - Best possible RMSE (using the optimal predictor)
"""

import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor

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

# Train a tiny MLPRegressor
# Using a small architecture: single hidden layer with 16 neurons
print("\nTraining tiny MLPRegressor (16 hidden units)...")
model = MLPRegressor(
    hidden_layer_sizes=(16,),
    max_iter=1000,
    random_state=42,
    early_stopping=False
)
model.fit(train_x, train_y)

print("Training complete!")

# Generate predictions on test set
print("\nGenerating predictions on test set...")
test_yhat = model.predict(test_x)

# Save predictions as float16 to match the other data files
output_path = data_dir / "mlp_test_yhat.npy"
np.save(output_path, test_yhat.astype(np.float16))
print(f"Predictions saved to: {output_path}")

# Calculate RMSE for MLP predictions (using float32 for accuracy)
mse_mlp = np.mean((test_yhat.astype(np.float32) - test_y.astype(np.float32)) ** 2)
rmse_mlp = np.sqrt(mse_mlp)
print(f"{'='*60}")
print(f"MLP Baseline RMSE: {rmse_mlp:.4f}")
print(f"{'='*60}")
print("This is a not-awful baseline score using a tiny MLP (16 hidden units).")

# Calculate best possible RMSE using the optimal predictor for this problem
print(f"\n{'='*60}")
print("Calculating best possible RMSE using optimal predictor...")
test_x_flat = test_x.flatten()
# The optimal predictor (minimizing expected MSE) for this data distribution
optimal_predictions = 0.5 * 10 * np.cos(test_x_flat)
mse_optimal = np.mean((optimal_predictions.astype(np.float32) - test_y.astype(np.float32)) ** 2)
rmse_optimal = np.sqrt(mse_optimal)
print(f"Best Possible RMSE (optimal predictor): {rmse_optimal:.4f}")
print(f"{'='*60}")
print("This is the essentially best-possible score (expected RMSE).")

# Print summary
print(f"\n{'='*60}")
print("SUMMARY:")
print(f"  MLP Baseline RMSE:    {rmse_mlp:.4f} (not-awful)")
print(f"  Best Possible RMSE:   {rmse_optimal:.4f} (optimal)")
print(f"  Difference:           {rmse_mlp - rmse_optimal:.4f}")
print(f"{'='*60}")
