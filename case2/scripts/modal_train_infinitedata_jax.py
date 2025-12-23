"""
Train infinite data solution using JAX on Modal.com with T4 GPU.

This script implements rectified flow matching (Liu et al. 2022) with infinite data generation
using JAX for GPU acceleration. Uses the linear interpolation path z_t = t*y + (1-t)*eps
with target velocity v = y - eps.

Architecture matches the reference implementation: (256, 128, 128, 64) with raw features only.

Training uses minibatched AdamW optimization with gradient clipping for stability.
The model is fully JIT-compiled including ODE integration for efficient GPU execution.

Outputs:
- training_loss.csv: Training loss over time (per step)
- energy_score.csv: Energy score over time (computed every 50 steps)
- scatter_samples.csv: 1000 conditional samples from the trained model

Note: The local entrypoint saves CSV files only (no plotting dependencies required locally).
For visualization, you can load the CSVs in your preferred plotting tool or use matplotlib locally.
"""

import modal

# ODE integration constants
ODE_NUM_STEPS = 100  # Number of Euler integration steps
ODE_DT = 1.0 / ODE_NUM_STEPS  # Time step size for integration from 0 to 1

# Create Modal app
app = modal.App("case2-infinitedata-jax")

# Define the image with JAX and required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]",
        "optax",
        "numpy",
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=30 * 60,  # 30 minute timeout as backstop
)
def train_model(duration_minutes=5, n_train_per_step=90000, learning_rate=0.0001, batch_size=4096):
    """
    Train rectified flow model using JAX with infinite data.
    
    Args:
        duration_minutes: How long to train (in minutes)
        n_train_per_step: Number of training samples to generate per step
        learning_rate: Learning rate for AdamW optimizer
        batch_size: Minibatch size for training
    
    Returns:
        Dictionary with training history and samples
    """
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import jax.lax as lax
    import optax
    import numpy as np
    import time
    
    # Verify GPU is available
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    backend = jax.default_backend()
    print(f"JAX backend: {backend}")
    
    # Hard check for GPU
    if backend != "gpu" and jax.devices()[0].platform != "gpu":
        raise RuntimeError(
            f"Expected GPU backend but got '{backend}'. "
            f"Device platform: {jax.devices()[0].platform}. "
            "This script requires GPU acceleration."
        )
    
    print(f"✓ GPU backend confirmed")
    print(f"Training for {duration_minutes} minutes with learning_rate={learning_rate}, batch_size={batch_size}")
    
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
        return x.squeeze()  # Return scalar instead of [1] array
    
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
        
        For each sample, draws n_t_per_sample independent random t values from Beta(2, 2) distribution.
        Beta(2, 2) concentrates samples around t=0.5 while still covering the full [0, 1] range.
        """
        n_samples = len(x_data)
        n_total = n_samples * n_t_per_sample
        
        # Replicate each sample n_t_per_sample times
        x_expanded = jnp.repeat(x_data, n_t_per_sample)
        y_expanded = jnp.repeat(y_data, n_t_per_sample)
        
        # Generate t values from Beta(2, 2) distribution for all samples
        # Beta(2, 2) has mean=0.5 and concentrates probability around the middle
        k_eps, k_t = random.split(key)
        t_values = random.beta(k_t, 2.0, 2.0, shape=(n_total,))
        
        # Generate random noise
        eps_values = random.normal(k_eps, (n_total,))
        
        # Compute z_t = y*t + (1-t)*eps (linear interpolation)
        zt_values = y_expanded * t_values + (1 - t_values) * eps_values
        
        # Target: y - eps (rectified flow velocity target)
        targets = y_expanded - eps_values
        
        # Create features (x, t, zt)
        features = jnp.stack([x_expanded, t_values, zt_values], axis=1)
        
        return features, targets
    
    @jit
    def loss_fn(params, features, targets):
        """MSE loss."""
        predictions = vmap(lambda x: mlp_forward(params, x))(features)
        return jnp.mean((predictions - targets) ** 2)
    
    def euler_integrate_batch(params, x_batch, z0_batch):
        """
        JAX-accelerated Euler ODE integration for a batch of initial conditions.
        
        Integrates from t=0 to t=1.0 using fixed timesteps.
        
        Args:
            params: Model parameters
            x_batch: Batch of x values (shape: [batch_size])
            z0_batch: Batch of initial z values (shape: [batch_size])
        
        Returns:
            Final z values after integration (shape: [batch_size])
        """
        def step_fn(z, t_idx):
            # Compute t from step index: t ranges from 0 to 1.0
            t = (t_idx + 1) * ODE_DT  # Start from ODE_DT, end at 1.0
            
            # Create feature array for batch: [x, t, z]
            # t needs to be broadcast to match batch size
            t_batch = jnp.full_like(z, t)
            features = jnp.stack([x_batch, t_batch, z], axis=1)
            
            # Compute velocity for entire batch
            v = vmap(lambda feat: mlp_forward(params, feat))(features)
            
            # Euler step: z_new = z + v * dt
            z_new = z + v * ODE_DT
            return z_new, None
        
        # Use lax.fori_loop for fast iteration (0 to ODE_NUM_STEPS-1)
        # This integrates from t=ODE_DT to t=1.0
        z_final, _ = lax.scan(step_fn, z0_batch, jnp.arange(ODE_NUM_STEPS))
        return z_final
    
    # JIT compile the integration for speed
    euler_integrate_batch_jit = jit(euler_integrate_batch)
    
    # Initialize model
    print("Initializing model...")
    layer_sizes = [input_dim] + hidden_layers + [output_dim]
    key, init_key = random.split(key)
    params = init_network_params(layer_sizes, init_key)
    
    # Initialize AdamW optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
        optax.adamw(learning_rate=learning_rate)
    )
    opt_state = optimizer.init(params)
    
    @jit
    def update_step(params, opt_state, features, targets):
        """Single optimization step using AdamW."""
        loss = loss_fn(params, features, targets)
        grads = grad(loss_fn)(params, features, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Generate initial data for computing standardization stats
    print("Generating initial data for standardization...")
    key, data_key = random.split(key)
    init_x, init_y = generate_training_data(1000, data_key)
    x_mean, x_std, y_mean, y_std = standardize_data(init_x, init_y)
    
    print(f"Standardization stats: x_mean={float(x_mean):.4f}, x_std={float(x_std):.4f}, y_mean={float(y_mean):.4f}, y_std={float(y_std):.4f}")
    
    # Training loop
    print(f"\nStarting training for {duration_minutes} minutes...")
    print(f"Step     Train Loss    Time Elapsed")
    print("-" * 50)
    
    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    halfway_time = start_time + (duration_minutes * 60) / 2
    learning_rate_halved = False
    
    step = 0
    train_losses = []
    energy_scores = []
    steps_recorded = []
    times_recorded = []
    energy_steps = []
    energy_times = []
    
    while time.time() < end_time:
        # Check if we've passed the halfway point and need to halve the learning rate
        if not learning_rate_halved and time.time() >= halfway_time:
            print(f"\n{'='*50}")
            print(f"Halfway point reached! Halving learning rate from {learning_rate} to {learning_rate/2}")
            print(f"{'='*50}\n")
            
            # Create new optimizer with halved learning rate
            new_learning_rate = learning_rate / 2
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=new_learning_rate)
            )
            # Reinitialize optimizer state with current params
            opt_state = optimizer.init(params)
            
            # Update the update_step function with the new optimizer
            @jit
            def update_step(params, opt_state, features, targets):
                """Single optimization step using AdamW."""
                loss = loss_fn(params, features, targets)
                grads = grad(loss_fn)(params, features, targets)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss
            
            learning_rate_halved = True
        
        # Generate fresh training data
        key, data_key = random.split(key)
        train_x, train_y = generate_training_data(n_train_per_step, data_key)
        train_x_scaled, train_y_scaled = transform_data(train_x, train_y, x_mean, x_std, y_mean, y_std)
        
        # Generate flow batch
        key, batch_key = random.split(key)
        features, targets = generate_flow_batch(train_x_scaled, train_y_scaled, 3, batch_key)
        
        # Shuffle
        key, shuffle_key = random.split(key)
        perm = random.permutation(shuffle_key, len(features))
        features = features[perm]
        targets = targets[perm]
        
        # Minibatched training: split into chunks
        n_batches = len(features) // batch_size
        batch_losses = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_features = features[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Update model
            params, opt_state, batch_loss = update_step(params, opt_state, batch_features, batch_targets)
            batch_losses.append(float(batch_loss))
        
        # Average loss across minibatches for this step
        train_loss = np.mean(batch_losses)
        step += 1
        
        # Record every 10 steps
        if step % 10 == 0:
            elapsed = time.time() - start_time
            train_losses.append(train_loss)
            steps_recorded.append(step)
            times_recorded.append(elapsed)
            
            print(f"{step:5d}    {train_loss:10.6f}    {elapsed:6.1f}s")
            
            # Calculate energy score every 50 steps
            if step % 50 == 0:
                # Generate validation data
                key, val_key = random.split(key)
                val_x, val_y = generate_training_data(100, val_key)
                val_x_scaled, _ = transform_data(val_x, val_y, x_mean, x_std, y_mean, y_std)
                
                # Generate samples using JAX-accelerated Euler integration
                # Generate 2 samples per validation point
                key, sample_key = random.split(key)
                z0_keys = random.split(sample_key, 200)  # 100 x values * 2 samples each
                
                # Create batches: repeat each x twice for 2 samples
                val_x_expanded = jnp.repeat(val_x_scaled, 2)
                z0_samples = jnp.array([random.normal(k, ()) for k in z0_keys])
                
                # Batch integrate all samples at once (fully GPU-accelerated)
                val_samples_scaled = euler_integrate_batch_jit(params, val_x_expanded, z0_samples)
                
                # Reshape to (n_val, 2)
                val_samples_scaled = val_samples_scaled.reshape(len(val_x), 2)
                
                # Transform back to original scale
                val_samples_orig = inverse_transform_y(val_samples_scaled, y_mean, y_std)
                
                # Calculate energy score
                # Energy = E[|Y - Y'|] - 0.5 * E[|Y' - Y''|]
                # where Y is true, Y' and Y'' are two independent samples
                sum_d1 = jnp.sum(jnp.abs(val_y - val_samples_orig[:, 0]))
                sum_d2 = jnp.sum(jnp.abs(val_y - val_samples_orig[:, 1]))
                sum_ds = jnp.sum(jnp.abs(val_samples_orig[:, 0] - val_samples_orig[:, 1]))
                
                val_energy = (sum_d1 + sum_d2) / (2 * len(val_y)) - 0.5 * sum_ds / len(val_y)
                energy_scores.append(float(val_energy))
                energy_steps.append(step)
                energy_times.append(elapsed)
                
                print(f"         Energy Score: {float(val_energy):.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Generate final samples for 1000 points using JAX-accelerated sampling
    print("\nGenerating 1000 samples for scatter plot...")
    key, sample_key = random.split(key)
    scatter_x, scatter_y = generate_training_data(1000, sample_key)
    scatter_x_scaled, _ = transform_data(scatter_x, scatter_y, x_mean, x_std, y_mean, y_std)
    
    # Generate random initial conditions for all samples at once
    key, z0_key = random.split(key)
    z0_samples = random.normal(z0_key, (1000,))
    
    # Batch integrate all 1000 samples at once (fully GPU-accelerated)
    print("  Running batch ODE integration on GPU...")
    scatter_samples_scaled = euler_integrate_batch_jit(params, scatter_x_scaled, z0_samples)
    scatter_samples_orig = inverse_transform_y(scatter_samples_scaled, y_mean, y_std)
    print("  ✓ Sampling complete!")
    
    return {
        'steps': steps_recorded,
        'train_losses': train_losses,
        'energy_scores': energy_scores,
        'energy_steps': energy_steps,
        'energy_times': energy_times,
        'times': times_recorded,
        'scatter_x': np.array(scatter_x),
        'scatter_y': np.array(scatter_y),
        'scatter_samples': np.array(scatter_samples_orig),
        'total_time': total_time,
    }


@app.local_entrypoint()
def main(duration_minutes: int = 7):
    """
    Main entrypoint for running training on Modal.
    
    Args:
        duration_minutes: How long to train (in minutes)
    """
    import csv
    from pathlib import Path
    
    print(f"Starting training on Modal with T4 GPU for {duration_minutes} minutes...")
    
    # Run training on Modal
    result = train_model.remote(duration_minutes=duration_minutes)
    
    # Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "modal_outputs"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving results to {output_dir}...")
    
    # Save CSVs using standard library csv module
    # Training loss CSV
    with open(output_dir / "training_loss.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'time_seconds'])
        for step, loss, time_val in zip(result['steps'], result['train_losses'], result['times']):
            writer.writerow([step, loss, time_val])
    
    # Energy score CSV (only every 50 steps)
    if len(result['energy_scores']) > 0:
        with open(output_dir / "energy_score.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'energy_score', 'time_seconds'])
            for step, score, time_val in zip(result['energy_steps'], result['energy_scores'], result['energy_times']):
                writer.writerow([step, score, time_val])
    
    # Scatter samples CSV
    with open(output_dir / "scatter_samples.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y_true', 'y_sampled'])
        for x, y_true, y_sampled in zip(result['scatter_x'], result['scatter_y'], result['scatter_samples']):
            writer.writerow([x, y_true, y_sampled])
    
    print("\nAll outputs saved successfully!")
    print(f"  - training_loss.csv")
    if len(result['energy_scores']) > 0:
        print(f"  - energy_score.csv")
    print(f"  - scatter_samples.csv")
    print(f"\nTotal training time: {result['total_time']:.2f} seconds ({result['total_time']/60:.2f} minutes)")
    print(f"\nNote: CSV files saved. You can visualize them with pandas/matplotlib:")
    print(f"  import pandas as pd; import matplotlib.pyplot as plt")
    print(f"  df = pd.read_csv('{output_dir / 'training_loss.csv'}')")
    print(f"  plt.plot(df['time_seconds'], df['train_loss']); plt.show()")
