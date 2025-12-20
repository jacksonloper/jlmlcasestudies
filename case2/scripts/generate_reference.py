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
    """
    X_batch = []
    y_batch = []
    
    for i in range(len(x_data)):
        x_i = x_data[i]
        y_i = y_data[i]
        
        # Exactly 3 t values: beginning (0), ending (1), and random middle
        t_values = [0.0, 1.0, np.random.uniform(0.0, 1.0)]
        
        for t in t_values:
            # Generate random standard normal noise
            eps = np.random.randn()
            
            # Compute z_t = y*t + (1-t)*eps
            zt = y_i * t + (1 - t) * eps
            
            # Create features from Fourier embeddings + raw values
            features = create_features(x_i, t, zt)
            
            # Target is y - eps (velocity field)
            target = y_i - eps
            
            X_batch.append(features)
            y_batch.append(target)
    
    return np.array(X_batch), np.array(y_batch)

# Step 2: Train MLP using partial_fit with fresh t samples each epoch
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

best_val_score = -np.inf
patience = 30
no_improve_count = 0

# Track errors for plotting
train_errors = []
val_errors = []
epochs_recorded = []

print(f"Epoch    Train Loss    Val Score")
print("-" * 40)

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
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val_fixed, y_val_fixed)
        
        # Store for plotting (convert R² to MSE-like loss)
        train_errors.append(-train_score)
        val_errors.append(-val_score)
        epochs_recorded.append(epoch + 1)
        
        print(f"{epoch+1:5d}    {-train_score:10.4f}    {val_score:9.4f}")
        
        # Early stopping
        if val_score > best_val_score:
            best_val_score = val_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience // 10:
            print(f"Early stopping at epoch {epoch+1}")
            break

training_end_time = time.time()
training_time = training_end_time - training_start_time

print(f"Training complete! Time: {training_time:.2f} seconds")

# Step 3: Generate samples using improved ODE solver
print("\nGenerating samples using ODE integration...")

def velocity_field(t, z, x_val):
    """Velocity field for the ODE: dz/dt = v(x, t, z)"""
    # Ensure z is scalar
    z_scalar = float(np.atleast_1d(z).flatten()[0])
    
    # Create features
    features = create_features(x_val, t, z_scalar)
    
    # Predict velocity (y - eps)
    v = model.predict(features.reshape(1, -1))[0]
    
    return [v]

def generate_sample(x_val):
    """Generate a sample by integrating from z_0 ~ N(0,1) to z_1 ~ y"""
    # Initial condition: z_0 ~ N(0,1)
    z0 = [np.random.randn()]
    
    # Time span for integration (from 0 to 1)
    t_span = (0, 1)
    
    # Solve ODE using RK45 with tighter tolerances
    try:
        solution = solve_ivp(
            velocity_field,
            t_span,
            z0,
            args=(x_val,),
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

# Generate two samples per test point (in scaled space)
samples_scaled = np.zeros((len(test_x_scaled), 2), dtype=np.float32)

for i in range(len(test_x_scaled)):
    if (i + 1) % 10 == 0:
        print(f"  Generating samples {i+1}/{len(test_x_scaled)}...")
    
    x_val = test_x_scaled[i]
    
    # Generate two independent samples
    samples_scaled[i, 0] = generate_sample(x_val)
    samples_scaled[i, 1] = generate_sample(x_val)

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
    'train_errors': [float(x) for x in train_errors],
    'val_errors': [float(x) for x in val_errors],
    'training_time': float(training_time),
    'hardware': platform.processor() or platform.machine(),
    'energy_score': float(energy_score)
}
history_path = data_dir / "reference_training_history.json"
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"Training history saved to: {history_path}")

# Also save as "optimal_samples.npy" for backwards compatibility with frontend
output_path_compat = data_dir / "optimal_samples.npy"
np.save(output_path_compat, samples_float16)
print(f"Also saved as: {output_path_compat} (for frontend compatibility)")
