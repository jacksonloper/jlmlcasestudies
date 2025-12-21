"""
Train infinite data solution using JAX on Modal.com with T4 GPU.

This script runs rectified flow matching with infinite data generation using JAX
for GPU acceleration. It runs for a fixed time duration (10 minutes by default)
without early stopping since we have infinite data.

Architecture matches the reference implementation: (256, 128, 128, 64) with raw features only.

Outputs:
- training_loss.csv: Training loss over time
- energy_score.csv: Energy score over time  
- scatter_samples.csv: 1000 samples from the trained model
- training_loss.png: Plot of training loss
- energy_score.png: Plot of energy score
- scatter_plot.png: Scatterplot of samples
"""

import modal
import os

# ODE integration constants
ODE_NUM_STEPS = 100  # Number of Euler integration steps
ODE_DT = 0.01  # Time step size (1.0 / ODE_NUM_STEPS)

# Create Modal app
app = modal.App("case2-infinitedata-jax")

# Define the image with JAX and required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]",
        "numpy",
        "scipy",
        "matplotlib",
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,  # 1 hour max timeout
)
def train_model(duration_minutes=10, n_train_per_epoch=900, learning_rate=0.001):
    """
    Train rectified flow model using JAX with infinite data.
    
    Args:
        duration_minutes: How long to train (in minutes)
        n_train_per_epoch: Number of training samples per epoch
        learning_rate: Learning rate for optimizer
    
    Returns:
        Dictionary with training history and samples
    """
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import numpy as np
    from scipy.integrate import solve_ivp
    import time
    
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Training for {duration_minutes} minutes with learning_rate={learning_rate}")
    
    # Set random seeds
    key = random.PRNGKey(42)
    np.random.seed(42)
    
    # Architecture: (256, 128, 128, 64) matching reference
    hidden_layers = [256, 128, 128, 64]
    input_dim = 3  # x, t, zt (raw features only)
    output_dim = 1  # velocity field
    
    def init_network_params(layer_sizes, key):
        """Initialize MLP parameters with Xavier initialization."""
        params = []
        keys = random.split(key, len(layer_sizes))
        
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            k1, k2 = random.split(keys[i])
            # Xavier initialization
            w = random.normal(k1, (n_in, n_out)) * jnp.sqrt(2.0 / (n_in + n_out))
            b = jnp.zeros(n_out)
            params.append((w, b))
        
        return params
    
    def mlp_forward(params, x):
        """Forward pass through MLP with ReLU activations."""
        for i, (w, b) in enumerate(params[:-1]):
            x = jnp.dot(x, w) + b
            x = jax.nn.relu(x)
        
        # Final layer (no activation)
        w, b = params[-1]
        x = jnp.dot(x, w) + b
        return x.flatten()
    
    def generate_training_data(n_samples, key):
        """
        Generate training data from the true generative model.
        
        - x ~ N(4, 1)
        - y | x is mixture of N(10*cos(x), 1) and N(0, 1)
        """
        k1, k2, k3, k4 = random.split(key, 4)
        
        # Generate x values from N(4, 1)
        x = random.normal(k1, (n_samples,)) + 4.0
        
        # Generate y values as mixture
        # Flip coins for mixture component
        mixture_selector = random.uniform(k2, (n_samples,)) < 0.5
        
        # Component 1: N(10*cos(x), 1) - uses k3
        y1 = random.normal(k3, (n_samples,)) + 10.0 * jnp.cos(x)
        
        # Component 2: N(0, 1) - uses k4
        y2 = random.normal(k4, (n_samples,))
        
        # Select based on mixture
        y = jnp.where(mixture_selector, y1, y2)
        
        return x, y
    
    def standardize_data(x, y):
        """Compute standardization statistics."""
        x_mean = jnp.mean(x)
        x_std = jnp.std(x)
        y_mean = jnp.mean(y)
        y_std = jnp.std(y)
        return x_mean, x_std, y_mean, y_std
    
    def transform_data(x, y, x_mean, x_std, y_mean, y_std):
        """Apply standardization."""
        x_scaled = (x - x_mean) / (x_std + 1e-8)
        y_scaled = (y - y_mean) / (y_std + 1e-8)
        return x_scaled, y_scaled
    
    def inverse_transform_y(y_scaled, y_mean, y_std):
        """Inverse transform y."""
        return y_scaled * y_std + y_mean
    
    def generate_flow_batch(x_data, y_data, n_t_per_sample, key):
        """
        Generate flow training batch.
        
        For each sample, uses 3 t values: t=0, t=1, t~random
        """
        n_samples = len(x_data)
        n_total = n_samples * n_t_per_sample
        
        # Replicate each sample n_t_per_sample times
        x_expanded = jnp.repeat(x_data, n_t_per_sample)
        y_expanded = jnp.repeat(y_data, n_t_per_sample)
        
        # Generate t values: [0, 1, random, 0, 1, random, ...]
        t_values = jnp.zeros(n_total)
        t_values = t_values.at[1::n_t_per_sample].set(1.0)
        
        # Fill random t values
        key_parts = random.split(key, n_t_per_sample)
        k_eps = key_parts[0]
        for i in range(2, n_t_per_sample):
            k_t = key_parts[i]
            random_t = random.uniform(k_t, (n_samples,))
            t_values = t_values.at[i::n_t_per_sample].set(random_t)
        
        # Generate random noise
        eps_values = random.normal(k_eps, (n_total,))
        
        # Compute z_t = y*t + (1-t)*eps
        zt_values = y_expanded * t_values + (1 - t_values) * eps_values
        
        # Target: y - eps
        targets = y_expanded - eps_values
        
        # Create features (x, t, zt)
        features = jnp.stack([x_expanded, t_values, zt_values], axis=1)
        
        return features, targets
    
    @jit
    def loss_fn(params, features, targets):
        """MSE loss."""
        predictions = vmap(lambda x: mlp_forward(params, x))(features)
        return jnp.mean((predictions - targets) ** 2)
    
    @jit
    def update_step(params, features, targets, learning_rate):
        """Single gradient descent step using vanilla SGD."""
        loss = loss_fn(params, features, targets)
        grads = grad(loss_fn)(params, features, targets)
        
        # Update parameters with vanilla SGD
        new_params = []
        for (w, b), (dw, db) in zip(params, grads):
            new_params.append((w - learning_rate * dw, b - learning_rate * db))
        
        return new_params, loss
    
    # Initialize model
    print("Initializing model...")
    layer_sizes = [input_dim] + hidden_layers + [output_dim]
    key, init_key = random.split(key)
    params = init_network_params(layer_sizes, init_key)
    
    # Generate initial data for computing standardization stats
    print("Generating initial data for standardization...")
    key, data_key = random.split(key)
    init_x, init_y = generate_training_data(1000, data_key)
    x_mean, x_std, y_mean, y_std = standardize_data(init_x, init_y)
    
    print(f"Standardization stats: x_mean={x_mean:.4f}, x_std={x_std:.4f}, y_mean={y_mean:.4f}, y_std={y_std:.4f}")
    
    # Training loop
    print(f"\nStarting training for {duration_minutes} minutes...")
    print(f"Epoch    Train Loss    Time Elapsed")
    print("-" * 50)
    
    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    
    epoch = 0
    train_losses = []
    energy_scores = []
    epochs_recorded = []
    times_recorded = []
    energy_epochs = []
    energy_times = []
    
    while time.time() < end_time:
        # Generate fresh training data
        key, data_key = random.split(key)
        train_x, train_y = generate_training_data(n_train_per_epoch, data_key)
        train_x_scaled, train_y_scaled = transform_data(train_x, train_y, x_mean, x_std, y_mean, y_std)
        
        # Generate flow batch
        key, batch_key = random.split(key)
        features, targets = generate_flow_batch(train_x_scaled, train_y_scaled, 3, batch_key)
        
        # Shuffle
        key, shuffle_key = random.split(key)
        perm = random.permutation(shuffle_key, len(features))
        features = features[perm]
        targets = targets[perm]
        
        # Update model
        params, train_loss = update_step(params, features, targets, learning_rate)
        
        epoch += 1
        
        # Record every 10 epochs
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            train_losses.append(float(train_loss))
            epochs_recorded.append(epoch)
            times_recorded.append(elapsed)
            
            print(f"{epoch:5d}    {train_loss:10.6f}    {elapsed:6.1f}s")
            
            # Calculate energy score every 50 epochs
            if epoch % 50 == 0:
                # Generate validation data
                key, val_key = random.split(key)
                val_x, val_y = generate_training_data(100, val_key)
                val_x_scaled, _ = transform_data(val_x, val_y, x_mean, x_std, y_mean, y_std)
                
                # Generate samples using ODE integration (simplified for JAX)
                # For energy score, we'll use a simpler Euler integration
                # Note: This is faster than RK45 but may be less accurate
                val_samples = []
                for i in range(len(val_x_scaled)):
                    x_val = val_x_scaled[i]
                    samples_for_x = []
                    
                    for _ in range(2):  # 2 samples per x
                        key, sample_key = random.split(key)
                        z0 = random.normal(sample_key, ())
                        
                        # Euler integration with fixed step size
                        z = z0
                        for step in range(ODE_NUM_STEPS):
                            t = step * ODE_DT
                            features_ode = jnp.array([x_val, t, z])
                            v = mlp_forward(params, features_ode)
                            z = z + v * ODE_DT
                        
                        samples_for_x.append(z)
                    
                    val_samples.append(samples_for_x)
                
                val_samples = jnp.array(val_samples)  # shape: (n_val, 2)
                
                # Transform back to original scale
                val_samples_orig = inverse_transform_y(val_samples, y_mean, y_std)
                
                # Calculate energy score
                sum_d1 = jnp.sum(jnp.abs(val_y - val_samples_orig[:, 0]))
                sum_d2 = jnp.sum(jnp.abs(val_y - val_samples_orig[:, 1]))
                sum_ds = jnp.sum(jnp.abs(val_samples_orig[:, 0] - val_samples_orig[:, 1]))
                
                val_energy = (sum_d1 + sum_d2) / (2 * len(val_y)) - 0.5 * sum_ds / len(val_y)
                energy_scores.append(float(val_energy))
                energy_epochs.append(epoch)
                energy_times.append(elapsed)
                
                print(f"         Energy Score: {val_energy:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Generate final samples for 1000 points
    print("\nGenerating 1000 samples for scatter plot...")
    key, sample_key = random.split(key)
    scatter_x, scatter_y = generate_training_data(1000, sample_key)
    scatter_x_scaled, _ = transform_data(scatter_x, scatter_y, x_mean, x_std, y_mean, y_std)
    
    scatter_samples = []
    for i in range(len(scatter_x_scaled)):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/1000 samples...")
        
        x_val = scatter_x_scaled[i]
        key, sample_key = random.split(key)
        z0 = random.normal(sample_key, ())
        
        # Euler integration with fixed step size
        z = z0
        for step in range(ODE_NUM_STEPS):
            t = step * ODE_DT
            features_ode = jnp.array([x_val, t, z])
            v = mlp_forward(params, features_ode)
            z = z + v * ODE_DT
        
        scatter_samples.append(z)
    
    scatter_samples = jnp.array(scatter_samples)
    scatter_samples_orig = inverse_transform_y(scatter_samples, y_mean, y_std)
    
    return {
        'epochs': epochs_recorded,
        'train_losses': train_losses,
        'energy_scores': energy_scores,
        'energy_epochs': energy_epochs,
        'energy_times': energy_times,
        'times': times_recorded,
        'scatter_x': np.array(scatter_x),
        'scatter_y': np.array(scatter_y),
        'scatter_samples': np.array(scatter_samples_orig),
        'total_time': total_time,
    }


@app.local_entrypoint()
def main(duration_minutes: int = 10):
    """
    Main entrypoint for running training on Modal.
    
    Args:
        duration_minutes: How long to train (in minutes)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    print(f"Starting training on Modal with T4 GPU for {duration_minutes} minutes...")
    
    # Run training on Modal
    result = train_model.remote(duration_minutes=duration_minutes)
    
    # Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "modal_outputs"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving results to {output_dir}...")
    
    # Save CSVs
    # Training loss CSV
    train_loss_data = np.column_stack([result['epochs'], result['train_losses'], result['times']])
    np.savetxt(
        output_dir / "training_loss.csv",
        train_loss_data,
        delimiter=",",
        header="epoch,train_loss,time_seconds",
        comments=""
    )
    
    # Energy score CSV (only every 50 epochs)
    if len(result['energy_scores']) > 0:
        energy_data = np.column_stack([result['energy_epochs'], result['energy_scores'], result['energy_times']])
        np.savetxt(
            output_dir / "energy_score.csv",
            energy_data,
            delimiter=",",
            header="epoch,energy_score,time_seconds",
            comments=""
        )
    
    # Scatter samples CSV
    scatter_data = np.column_stack([result['scatter_x'], result['scatter_y'], result['scatter_samples']])
    np.savetxt(
        output_dir / "scatter_samples.csv",
        scatter_data,
        delimiter=",",
        header="x,y_true,y_sampled",
        comments=""
    )
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Training loss plot
    axes[0].plot(result['times'], result['train_losses'], 'b-', linewidth=2)
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Over Time', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Energy score plot
    if len(result['energy_scores']) > 0:
        axes[1].plot(result['energy_times'], result['energy_scores'], 'r-', linewidth=2)
        axes[1].set_xlabel('Time (seconds)', fontsize=12)
        axes[1].set_ylabel('Energy Score', fontsize=12)
        axes[1].set_title('Energy Score Over Time', fontsize=14)
        axes[1].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[2].scatter(result['scatter_x'], result['scatter_samples'], alpha=0.3, s=20, label='Model Samples')
    axes[2].scatter(result['scatter_x'], result['scatter_y'], alpha=0.3, s=20, label='Ground Truth', color='red')
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('y', fontsize=12)
    axes[2].set_title('1000 Samples: Model vs Ground Truth', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_plots.png", dpi=150, bbox_inches='tight')
    print(f"Combined plot saved to {output_dir / 'training_plots.png'}")
    
    # Also save individual plots
    plt.figure(figsize=(8, 6))
    plt.plot(result['times'], result['train_losses'], 'b-', linewidth=2)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    if len(result['energy_scores']) > 0:
        plt.figure(figsize=(8, 6))
        plt.plot(result['energy_times'], result['energy_scores'], 'r-', linewidth=2)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Energy Score', fontsize=12)
        plt.title('Energy Score Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "energy_score.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(result['scatter_x'], result['scatter_samples'], alpha=0.3, s=20, label='Model Samples')
    plt.scatter(result['scatter_x'], result['scatter_y'], alpha=0.3, s=20, label='Ground Truth', color='red')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('1000 Samples: Model vs Ground Truth', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "scatter_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nAll outputs saved successfully!")
    print(f"  - training_loss.csv")
    print(f"  - energy_score.csv")
    print(f"  - scatter_samples.csv")
    print(f"  - training_loss.png")
    print(f"  - energy_score.png")
    print(f"  - scatter_plot.png")
    print(f"  - training_plots.png (combined)")
    print(f"\nTotal training time: {result['total_time']:.2f} seconds ({result['total_time']/60:.2f} minutes)")
