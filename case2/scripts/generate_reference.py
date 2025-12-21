"""
Generate reference solution for Case Study 2: Distribution Sampling using Rectified Flow Matching

This script uses improved Rectified Flow Matching with iterative training:
1. Uses partial_fit to train on fresh random t, eps samples each epoch
2. For each sample, uses exactly 3 t values: t=0 (beginning), t=1 (ending), t=random (middle)
3. Train MLP to predict y-eps from raw features of (x, t, z_t)
   where z_t = y*t + (1-t)*eps
4. Use scipy solve_ivp with N(0,1) initial conditions to generate samples

This is a reference solution that demonstrates how to learn the conditional distribution
without knowing the true mixture structure.

Outputs:
- case2/data/reference_samples.npy: 100x2 matrix (two samples per test point)
"""

import numpy as np
from pathlib import Path
import time
import platform
import json
from model import RectifiedFlowModel, create_raw_feature_extractor, calculate_energy_score

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
train_x = train_data[:, 0]
train_y = train_data[:, 1]

print(f"Training data shape: {train_data.shape}")

# Load test data from dataset1
print("\nLoading test data from dataset1...")
test_x = np.load(dataset_dir / "test_x.npy")
test_y = np.load(dataset_dir / "test_y.npy")

print(f"Test X shape: {test_x.shape}, Test Y shape: {test_y.shape}")

# Create model with raw features only (no Fourier)
print("\nSetting up rectified flow model with raw features (no Fourier)...")
print(f"Hardware: {platform.processor() or platform.machine()} ({platform.system()})")

feature_extractor = create_raw_feature_extractor()
model = RectifiedFlowModel(
    feature_extractor=feature_extractor,
    learning_rate=0.001,
    random_state=42
)

print(f"Architecture: {model.hidden_layers}")

# Train the model
print("\nTraining MLP with partial_fit (fresh t samples each epoch)...")
print("Using 3 t values per sample: t=0 (beginning), t=1 (ending), t=random (middle)")

training_start_time = time.time()

training_history = model.train(
    train_x=train_x,
    train_y=train_y,
    n_epochs=300,
    n_t_per_sample=3,
    val_split=0.1,
    patience=30,
    verbose=True
)

training_end_time = time.time()
training_time = training_end_time - training_start_time

print(f"Training complete! Time: {training_time:.2f} seconds")

# Generate samples for test set
print("\nGenerating samples using ODE integration...")
samples = model.predict_samples(test_x, n_samples=2, verbose=True)

# Calculate energy score
print("\nCalculating energy score...")
energy_score = calculate_energy_score(test_y, samples)

print(f"{'='*60}")
print(f"Rectified Flow Energy Score: {energy_score:.4f}")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Hardware: {platform.processor() or platform.machine()}")
print(f"{'='*60}")
print("This uses rectified flow matching to learn the conditional distribution.")
print("The model learns to transport samples from N(0,1) to the target distribution.")

# Convert to float16 ONLY for saving
samples_float16 = samples.astype(np.float16)

# Save samples
output_path = data_dir / "reference_samples.npy"
np.save(output_path, samples_float16)
print(f"\nReference samples saved to: {output_path}")

# Add additional fields to training history
training_history['training_time'] = float(training_time)
training_history['hardware'] = platform.processor() or platform.machine()
training_history['final_energy_score'] = float(energy_score)
history_path = data_dir / "reference_training_history.json"
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"Training history saved to: {history_path}")

# Also save as "optimal_samples.npy" for backwards compatibility with frontend
output_path_compat = data_dir / "optimal_samples.npy"
np.save(output_path_compat, samples_float16)
print(f"Also saved as: {output_path_compat} (for frontend compatibility)")
