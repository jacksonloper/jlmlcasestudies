"""
Generate reference solution for Case Study 2: Distribution Sampling using Rectified Flow Matching

This script uses Rectified Flow Matching to generate samples:
1. Generate 10 t values per training datapoint
2. For each t, generate random standard normal eps
3. Train MLP to predict y-eps from Fourier embeddings of (x, t, y*t + (1-t)*eps)
4. Use scipy odeint with N(0,1) initial conditions to generate samples

This is the reference solution that demonstrates how to learn the conditional distribution
without knowing the true mixture structure.

Outputs:
- case2/data/optimal_samples.npy: 100x2 matrix (two samples per test point)
"""

import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from scipy.integrate import odeint

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

def fourier_embedding(x, n_freqs=10):
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
    """Create Fourier embeddings of (x, t, zt)"""
    # Create Fourier embeddings for each component (each returns a flat array)
    x_emb = fourier_embedding(x, n_freqs=5)
    t_emb = fourier_embedding(t, n_freqs=5)
    zt_emb = fourier_embedding(zt, n_freqs=5)
    
    # Concatenate all features
    return np.concatenate([x_emb, t_emb, zt_emb])

# Step 1: Generate training data for rectified flow
print("\nGenerating training data for rectified flow...")
n_t_samples = 10  # Number of t values per datapoint
n_train = len(train_x)

# Preallocate arrays
X_flow = []
y_flow = []

for i in range(n_train):
    x_i = train_x[i]
    y_i = train_y[i]
    
    # Generate t values
    t_values = np.random.uniform(0, 1, n_t_samples)
    
    for t in t_values:
        # Generate random standard normal noise
        eps = np.random.randn()
        
        # Compute z_t = y*t + (1-t)*eps
        zt = y_i * t + (1 - t) * eps
        
        # Create features from Fourier embeddings
        features = create_features(x_i, t, zt)
        
        # Target is y - eps (velocity field)
        target = y_i - eps
        
        X_flow.append(features)
        y_flow.append(target)

X_flow = np.array(X_flow)
y_flow = np.array(y_flow)

print(f"Flow training data shape: X={X_flow.shape}, y={y_flow.shape}")

# Step 2: Train MLP to predict velocity field
print("\nTraining MLP for rectified flow...")
model = MLPRegressor(
    hidden_layer_sizes=(128, 128),
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
model.fit(X_flow, y_flow)

print("Training complete!")

# Step 3: Generate samples using ODE solver
print("\nGenerating samples using ODE integration...")

def velocity_field(z, t, x_val, model):
    """Velocity field for the ODE: dz/dt = v(x, t, z)"""
    # Create features
    features = create_features(x_val, t, z)
    # Predict velocity (y - eps)
    v = model.predict(features.reshape(1, -1))[0]
    return v

def generate_sample(x_val, model):
    """Generate a sample by integrating from z_0 ~ N(0,1) to z_1 ~ y"""
    # Initial condition: z_0 ~ N(0,1)
    z0 = np.random.randn()
    
    # Time points for integration (from 0 to 1)
    t_eval = np.linspace(0, 1, 20)
    
    # Solve ODE
    try:
        solution = odeint(velocity_field, z0, t_eval, args=(x_val, model))
        # Return final value z_1
        return solution[-1, 0]
    except (ValueError, RuntimeError) as e:
        # Fallback: return z0 if ODE fails
        print(f"Warning: ODE integration failed with {type(e).__name__}, using fallback")
        return z0

# Generate two samples per test point
samples = np.zeros((len(test_x), 2), dtype=np.float32)

for i in range(len(test_x)):
    if (i + 1) % 10 == 0:
        print(f"  Generating samples {i+1}/{len(test_x)}...")
    
    x_val = test_x[i]
    
    # Generate two independent samples
    samples[i, 0] = generate_sample(x_val, model)
    samples[i, 1] = generate_sample(x_val, model)

# Convert to float16 to match other files
samples = samples.astype(np.float16)

# Save samples
output_path = data_dir / "optimal_samples.npy"
np.save(output_path, samples)
print(f"\nOptimal samples saved to: {output_path}")

# Calculate energy score
print("\nCalculating energy score...")
sum_dist1 = np.sum(np.abs(test_y - samples[:, 0]))
sum_dist2 = np.sum(np.abs(test_y - samples[:, 1]))
sum_dist_samples = np.sum(np.abs(samples[:, 0] - samples[:, 1]))

energy_score = (sum_dist1 + sum_dist2) / (2 * len(test_y)) - 0.5 * sum_dist_samples / len(test_y)

print(f"{'='*60}")
print(f"Rectified Flow Energy Score: {energy_score:.4f}")
print(f"{'='*60}")
print("This uses rectified flow matching to learn the conditional distribution.")
print("The model learns to transport samples from N(0,1) to the target distribution.")

