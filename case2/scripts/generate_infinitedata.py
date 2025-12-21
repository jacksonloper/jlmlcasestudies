"""
Generate infinite data solution for Case Study 2: Distribution Sampling using Rectified Flow Matching

This script trains on infinite data by generating fresh samples from the true generative model
each epoch. Unlike the reference solution that uses finite data from dataset1:
1. Generates fresh training data each epoch from the true distribution: x ~ N(4, 1), 
   y|x is mixture of N(10*cos(x), 1) and N(0, 1)
2. Uses raw features only (NO Fourier embeddings)
3. Uses a larger architecture with an extra 256-neuron layer at the beginning
4. Uses partial_fit to train on fresh random t, eps samples each epoch
5. Uses scipy solve_ivp with N(0,1) initial conditions to generate samples

Outputs:
- case2/data/infinitedata_samples.npy: 100x2 matrix (two samples per test point)
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

# Load test data from dataset1 (for evaluation only)
print("Loading test data from dataset1...")
test_x = np.load(dataset_dir / "test_x.npy")
test_y = np.load(dataset_dir / "test_y.npy")

print(f"Test X shape: {test_x.shape}, Test Y shape: {test_y.shape}")

def generate_training_data(n_samples, random_state=None):
    """
    Generate training data from the true generative model.
    
    - x ~ N(4, 1)
    - y | x is an equal parts mixture of N(10*cos(x), 1) and N(0, 1)
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed (optional)
        
    Returns:
        train_x, train_y: Arrays of x and y values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate x values from N(4, 1)
    x = np.random.normal(4, 1, n_samples)
    
    # Generate y values as mixture
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Equal parts mixture: flip a coin
        if np.random.rand() < 0.5:
            # From N(10*cos(x), 1)
            y[i] = np.random.normal(10 * np.cos(x[i]), 1)
        else:
            # From N(0, 1)
            y[i] = np.random.normal(0, 1)
    
    return x, y


# Create model with raw features only (no Fourier) and larger architecture
print("\nSetting up rectified flow model with raw features (no Fourier)...")
print("Architecture: (256, 128, 128, 64) - extra 256-neuron layer at beginning")
print(f"Hardware: {platform.processor() or platform.machine()} ({platform.system()})")

feature_extractor = create_raw_feature_extractor()
model = RectifiedFlowModel(
    hidden_layers=(256, 128, 128, 64),  # Extra 256 layer at beginning
    feature_extractor=feature_extractor,
    learning_rate=0.001,
    random_state=42
)

# Generate initial training data to fit scalers
print("\nGenerating initial training data from true generative model...")
initial_train_x, initial_train_y = generate_training_data(n_samples=1000, random_state=42)
print(f"Initial training data: {len(initial_train_x)} samples")
print(f"  X: mean={initial_train_x.mean():.2f}, std={initial_train_x.std():.2f}")
print(f"  Y: mean={initial_train_y.mean():.2f}, std={initial_train_y.std():.2f}")

# Fit scalers on initial data
model.fit_scalers(initial_train_x, initial_train_y)

# Now we'll do custom training loop that generates fresh data each epoch
print("\nTraining with infinite data (fresh samples each epoch)...")
print("Using 3 t values per sample: t=0 (beginning), t=1 (ending), t=random (middle)")

training_start_time = time.time()

# Training parameters
n_epochs = 300
n_t_per_sample = 3
n_train_per_epoch = 900  # Same size as finite data for fair comparison
n_val = 100  # Fixed validation set

# Generate a fixed validation set
val_x, val_y = generate_training_data(n_samples=n_val, random_state=999)
val_x_scaled = model.transform_x(val_x)
val_y_scaled = model.transform_y(val_y)

# Fixed validation batch for consistent scoring
np.random.seed(9999)
X_val_fixed, y_val_fixed = model.generate_flow_batch(val_x_scaled, val_y_scaled, n_t_per_sample)
np.random.seed(42)

# Initialize model with initial batch
print("\nInitializing model...")
from sklearn.neural_network import MLPRegressor

train_x_scaled = model.transform_x(initial_train_x[:100])
train_y_scaled = model.transform_y(initial_train_y[:100])
X_init, y_init = model.generate_flow_batch(train_x_scaled, train_y_scaled, n_t_per_sample)

model.model = MLPRegressor(
    hidden_layer_sizes=model.hidden_layers,
    random_state=model.random_state,
    learning_rate_init=model.learning_rate,
    solver='adam',
    max_iter=1
)
model.model.partial_fit(X_init, y_init)

best_val_loss = np.inf
patience = 30
no_improve_count = 0

# Track errors for history
train_mse = []
val_mse = []
val_energy_scores = []
epochs_recorded = []

print(f"Epoch    Train MSE    Val MSE    Val Energy Score")
print("-" * 60)

for epoch in range(n_epochs):
    # Generate fresh training data from the generative model
    epoch_seed = 42 + epoch  # Different seed each epoch for variety
    train_x_epoch, train_y_epoch = generate_training_data(n_samples=n_train_per_epoch, random_state=epoch_seed)
    
    # Scale the data
    train_x_scaled = model.transform_x(train_x_epoch)
    train_y_scaled = model.transform_y(train_y_epoch)
    
    # Generate flow batch with random t, eps
    X_train, y_train = model.generate_flow_batch(train_x_scaled, train_y_scaled, n_t_per_sample)
    
    # Shuffle and train
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    # Update model with this epoch's batch
    model.model.partial_fit(X_train, y_train)
    
    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Calculate MSE loss
        train_pred = model.model.predict(X_train)
        val_pred = model.model.predict(X_val_fixed)
        
        train_loss = np.mean((train_pred - y_train) ** 2)
        val_loss = np.mean((val_pred - y_val_fixed) ** 2)
        
        # Calculate energy score on validation set
        val_samples_scaled = np.zeros((len(val_x_scaled), 2), dtype=np.float32)
        for i in range(len(val_x_scaled)):
            val_samples_scaled[i, 0] = model.generate_sample(val_x_scaled[i])
            val_samples_scaled[i, 1] = model.generate_sample(val_x_scaled[i])
        
        # Transform back to original scale
        val_samples = model.inverse_transform_y(val_samples_scaled.flatten()).reshape(-1, 2)
        val_y_orig = val_y
        
        # Calculate energy score
        sum_d1 = np.sum(np.abs(val_y_orig - val_samples[:, 0]))
        sum_d2 = np.sum(np.abs(val_y_orig - val_samples[:, 1]))
        sum_ds = np.sum(np.abs(val_samples[:, 0] - val_samples[:, 1]))
        val_energy = (sum_d1 + sum_d2) / (2 * len(val_y_orig)) - 0.5 * sum_ds / len(val_y_orig)
        val_energy_scores.append(float(val_energy))
        
        # Store for history
        train_mse.append(float(train_loss))
        val_mse.append(float(val_loss))
        epochs_recorded.append(epoch + 1)
        
        print(f"{epoch+1:5d}    {train_loss:10.4f}    {val_loss:8.4f}    {val_energy:14.4f}")
        
        # Early stopping based on validation MSE
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience // 10:
            print(f"Early stopping at epoch {epoch+1}")
            break

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
print(f"Infinite Data Energy Score: {energy_score:.4f}")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Hardware: {platform.processor() or platform.machine()}")
print(f"{'='*60}")
print("This uses rectified flow matching trained on infinite data.")
print("Fresh samples are generated from the true generative model each epoch.")
print("Architecture: (256, 128, 128, 64) with raw features only (no Fourier).")

# Convert to float16 ONLY for saving
samples_float16 = samples.astype(np.float16)

# Save samples
output_path = data_dir / "infinitedata_samples.npy"
np.save(output_path, samples_float16)
print(f"\nInfinite data samples saved to: {output_path}")

# Save training history
training_history = {
    'epochs': epochs_recorded,
    'train_mse': train_mse,
    'val_mse': val_mse,
    'val_energy_scores': val_energy_scores,
    'training_time': float(training_time),
    'hardware': platform.processor() or platform.machine(),
    'final_energy_score': float(energy_score),
    'architecture': 'raw_features_only',
    'hidden_layers': list(model.hidden_layers),
    'training_data': 'infinite (generated fresh each epoch)'
}
history_path = data_dir / "infinitedata_training_history.json"
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"Training history saved to: {history_path}")
