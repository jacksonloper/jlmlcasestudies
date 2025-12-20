"""
Generate reference solution for Case Study 2: Distribution Sampling using Rectified Flow Matching

This script uses improved Rectified Flow Matching with iterative training:
1. Uses partial_fit to train on fresh random t, eps samples each epoch
2. For each sample, uses exactly 3 t values: t=0 (beginning), t=1 (ending), t=random (middle)
3. Train MLP to predict y-eps from Fourier embeddings + raw features of (x, t, z_t)
   where z_t = y*t + (1-t)*eps
4. Use scipy solve_ivp with N(0,1) initial conditions to generate samples

This is a reference solution that demonstrates how to learn the conditional distribution
without knowing the true mixture structure.

Outputs:
- case2/data/reference_samples.npy: 100x2 matrix (two samples per test point)
"""

import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
import time
import platform
import json

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

# Standardize inputs for better training
print("\nStandardizing inputs...")
x_scaler = StandardScaler()
y_scaler = StandardScaler()

train_x_scaled = x_scaler.fit_transform(train_x.reshape(-1, 1)).flatten()
train_y_scaled = y_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
test_x_scaled = x_scaler.transform(test_x.reshape(-1, 1)).flatten()

def fourier_embedding(x, n_freqs=5):
    """Create Fourier features for input x (scalar or array)"""
    x = np.atleast_1d(x).flatten()
    freqs = 2 ** np.arange(n_freqs)
    features = []
    for freq in freqs:
        features.append(np.sin(2 * np.pi * freq * x))
        features.append(np.cos(2 * np.pi * freq * x))
    # Flatten all features into a single array
    return np.concatenate([f.flatten() for f in features])

def create_features(x, t, zt):
    """Create Fourier embeddings + raw features of (x, t, zt)"""
    # Ensure scalar inputs
    x = float(np.atleast_1d(x).flatten()[0])
    t = float(np.atleast_1d(t).flatten()[0])
    zt = float(np.atleast_1d(zt).flatten()[0])
    
    # Create Fourier embeddings for each component
    x_emb = fourier_embedding(x, n_freqs=5)
    t_emb = fourier_embedding(t, n_freqs=5)
    zt_emb = fourier_embedding(zt, n_freqs=5)
    
    # Concatenate raw values + Fourier features
    return np.concatenate([[x, t, zt], x_emb, t_emb, zt_emb])

# Step 1: Setup for iterative training with partial_fit
print("\nSetting up rectified flow training with partial_fit...")
n_train = len(train_x_scaled)
n_epochs = 300  # Number of training epochs
n_t_per_sample = 3  # Exactly 3 t values per sample (beginning, end, random middle)

# Hold out 10% for validation
n_val = int(0.1 * n_train)
val_indices = np.random.choice(n_train, n_val, replace=False)
train_indices = np.array([i for i in range(n_train) if i not in val_indices])

val_x = train_x_scaled[val_indices]
val_y = train_y_scaled[val_indices]
train_x_sub = train_x_scaled[train_indices]
train_y_sub = train_y_scaled[train_indices]

print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

def generate_flow_batch(x_data, y_data, n_t_per_sample):
    """Generate a fresh batch of flow training data with 3 t values per sample:
    - t=0 (beginning)
    - t=1 (ending)  
    - t~random (middle)
    
    Vectorized implementation for efficiency.
    """
    n_samples = len(x_data)
    n_total = n_samples * n_t_per_sample
    
    # Vectorized: replicate each sample 3 times
    x_expanded = np.repeat(x_data, n_t_per_sample)  # shape: (n_total,)
    y_expanded = np.repeat(y_data, n_t_per_sample)  # shape: (n_total,)
    
    # Vectorized: generate t values for all samples at once
    # For each sample: [0.0, 1.0, random]
    t_values = np.zeros(n_total)
    t_values[1::3] = 1.0  # Every 2nd element in each group of 3
    t_values[2::3] = np.random.uniform(0.0, 1.0, n_samples)  # Every 3rd element
    
    # Vectorized: generate all random noise at once
    eps_values = np.random.randn(n_total)
    
    # Vectorized: compute all z_t values
    zt_values = y_expanded * t_values + (1 - t_values) * eps_values
    
    # Vectorized: compute all targets
    y_batch = y_expanded - eps_values
    
    # Create features (still needs loop due to Fourier embedding complexity)
    X_batch = []
    for i in range(n_total):
        features = create_features(x_expanded[i], t_values[i], zt_values[i])
        X_batch.append(features)
    
    return np.array(X_batch), y_batch

# Step 2: Define helper functions for sampling
def velocity_field(t, z, x_val, model):
    """Velocity field for the ODE: dz/dt = v(x, t, z)"""
    # Ensure z is scalar
    z_scalar = float(np.atleast_1d(z).flatten()[0])
    
    # Create features
    features = create_features(x_val, t, z_scalar)
    
    # Predict velocity (y - eps)
    v = model.predict(features.reshape(1, -1))[0]
    
    return [v]

def generate_sample(x_val, model):
    """Generate a sample by integrating from z_0 ~ N(0,1) to z_1 ~ y"""
    # Initial condition: z_0 ~ N(0,1)
    z0 = [np.random.randn()]
    
    # Time span for integration (from 0 to 1)
    t_span = (0, 1)
    
    # Solve ODE using RK45 with tighter tolerances
    try:
        solution = solve_ivp(
            lambda t, z: velocity_field(t, z, x_val, model),
            t_span,
            z0,
            method='RK45',
            rtol=1e-5,
            atol=1e-7,
            dense_output=True
        )
        
        if solution.success:
            # Return final value z_1
            return solution.y[0, -1]
        else:
            print(f"Warning: ODE integration failed, using fallback")
            return z0[0]
    except (ValueError, RuntimeError) as e:
        # Fallback: return z0 if ODE fails
        print(f"Warning: ODE integration failed with {type(e).__name__}, using fallback")
        return z0[0]

# Step 3: Train MLP using partial_fit with fresh t samples each epoch
print("\nTraining MLP with partial_fit (fresh t samples each epoch)...")
print("Using 3 t values per sample: t=0 (beginning), t=1 (ending), t=random (middle)")
print(f"Hardware: {platform.processor() or platform.machine()} ({platform.system()})")

training_start_time = time.time()

# Initialize model
model = MLPRegressor(
    hidden_layer_sizes=(128, 128, 64),
    random_state=42,
    learning_rate_init=0.001,
    solver='adam',
    max_iter=1  # Not used with partial_fit, but required
)

# Generate initial batch to initialize model structure
X_init, y_init = generate_flow_batch(train_x_sub[:100], train_y_sub[:100], n_t_per_sample)
model.partial_fit(X_init, y_init)

# Fixed validation batch for consistent scoring (use fixed seed)
np.random.seed(999)
X_val_fixed, y_val_fixed = generate_flow_batch(val_x, val_y, n_t_per_sample)
np.random.seed(42)  # Reset to main seed

best_val_loss = np.inf  # Track best validation MSE (lower is better)
patience = 30
no_improve_count = 0

# Track errors for plotting
train_mse = []
val_mse = []
val_energy_scores = []
epochs_recorded = []

print(f"Epoch    Train MSE    Val MSE    Val Energy Score")
print("-" * 60)

for epoch in range(n_epochs):
    # Generate fresh batch with new random t, eps for each training sample
    X_train, y_train = generate_flow_batch(train_x_sub, train_y_sub, n_t_per_sample)
    
    # Shuffle and train
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    # Update model with this epoch's batch
    model.partial_fit(X_train, y_train)
    
    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        # Calculate MSE loss
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val_fixed)
        
        train_loss = np.mean((train_pred - y_train) ** 2)
        val_loss = np.mean((val_pred - y_val_fixed) ** 2)
        
        # Calculate energy score on validation set by generating samples
        # Generate 2 samples per validation point
        val_samples_scaled = np.zeros((len(val_x), 2), dtype=np.float32)
        for i in range(len(val_x)):
            val_samples_scaled[i, 0] = generate_sample(val_x[i], model)
            val_samples_scaled[i, 1] = generate_sample(val_x[i], model)
        
        # Transform back to original scale
        val_samples = y_scaler.inverse_transform(val_samples_scaled).astype(np.float32)
        val_y_orig = y_scaler.inverse_transform(val_y.reshape(-1, 1)).flatten()
        
        # Calculate energy score
        sum_d1 = np.sum(np.abs(val_y_orig - val_samples[:, 0]))
        sum_d2 = np.sum(np.abs(val_y_orig - val_samples[:, 1]))
        sum_ds = np.sum(np.abs(val_samples[:, 0] - val_samples[:, 1]))
        val_energy = (sum_d1 + sum_d2) / (2 * len(val_y_orig)) - 0.5 * sum_ds / len(val_y_orig)
        val_energy_scores.append(float(val_energy))
        
        # Store for plotting
        train_mse.append(float(train_loss))
        val_mse.append(float(val_loss))
        epochs_recorded.append(epoch + 1)
        
        print(f"{epoch+1:5d}    {train_loss:10.4f}    {val_loss:8.4f}    {val_energy:14.4f}")
        
        # Early stopping based on validation MSE
        if val_loss < best_val_loss:  # Lower MSE is better
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

# Step 4: Generate samples using improved ODE solver
print("\nGenerating samples using ODE integration...")

# Generate two samples per test point (in scaled space)
samples_scaled = np.zeros((len(test_x_scaled), 2), dtype=np.float32)

for i in range(len(test_x_scaled)):
    if (i + 1) % 10 == 0:
        print(f"  Generating samples {i+1}/{len(test_x_scaled)}...")
    
    x_val = test_x_scaled[i]
    
    # Generate two independent samples
    samples_scaled[i, 0] = generate_sample(x_val, model)
    samples_scaled[i, 1] = generate_sample(x_val, model)

# Transform back to original scale
samples = y_scaler.inverse_transform(samples_scaled).astype(np.float32)

# Calculate energy score BEFORE converting to float16
print("\nCalculating energy score...")
sum_dist1 = np.sum(np.abs(test_y - samples[:, 0]))
sum_dist2 = np.sum(np.abs(test_y - samples[:, 1]))
sum_dist_samples = np.sum(np.abs(samples[:, 0] - samples[:, 1]))

energy_score = (sum_dist1 + sum_dist2) / (2 * len(test_y)) - 0.5 * sum_dist_samples / len(test_y)

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

# Save training history
training_history = {
    'epochs': epochs_recorded,
    'train_mse': train_mse,
    'val_mse': val_mse,
    'val_energy_scores': val_energy_scores,
    'training_time': float(training_time),
    'hardware': platform.processor() or platform.machine(),
    'final_energy_score': float(energy_score)
}
history_path = data_dir / "reference_training_history.json"
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"Training history saved to: {history_path}")

# Also save as "optimal_samples.npy" for backwards compatibility with frontend
output_path_compat = data_dir / "optimal_samples.npy"
np.save(output_path_compat, samples_float16)
print(f"Also saved as: {output_path_compat} (for frontend compatibility)")
