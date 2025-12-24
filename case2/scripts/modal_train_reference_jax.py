"""
Train reference solution using JAX on Modal.com with T4 GPU.

This script implements rectified flow matching (Liu et al. 2022) with finite training data
using JAX for GPU acceleration. Uses the linear interpolation path z_t = t*y + (1-t)*eps
with target velocity v = y - eps.

Architecture matches the infinite data implementation: (256, 128, 128, 64) with raw features only.

Training uses minibatched AdamW optimization with gradient clipping for stability.
The model is fully JIT-compiled including ODE integration for efficient GPU execution.

Key differences from infinite data:
- Uses finite 900 training samples from dataset1/data/train.npy
- Samples multiple t values per training point to achieve comparable batch size
- Since dataset is small, epochs are meaningless; instead trains for fixed wall clock time
- Logs at similar wall clock frequency as infinite data script

Outputs:
- reference_training_loss.csv: Training loss over time (per step)
- reference_energy_score.csv: Energy score over time (computed periodically)
- reference_scatter_samples.csv: 1000 conditional samples from the trained model

Note: The local entrypoint saves CSV files only (no plotting dependencies required locally).
For visualization, you can load the CSVs in your preferred plotting tool or use matplotlib locally.
"""

import modal

# Create Modal app
app = modal.App("case2-reference-jax")

# Define the image with JAX and required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]",
        "optax",
        "numpy",
        "diffrax",
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=30 * 60,  # 30 minute timeout as backstop
)
def train_model(train_x_list, train_y_list, test_x_list, test_y_list, duration_minutes=2, learning_rate=0.0001, batch_size=4096):
    """
    Train rectified flow model using JAX with finite data.
    
    Args:
        train_x_list: Training x values as list (will be converted to JAX array)
        train_y_list: Training y values as list (will be converted to JAX array)
        test_x_list: Held-out test x values for energy score computation
        test_y_list: Held-out test y values for energy score computation
        duration_minutes: How long to train (in minutes)
        learning_rate: Learning rate for AdamW optimizer
        batch_size: Minibatch size for training
    
    Returns:
        Dictionary with training history and samples
    """
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import optax
    import diffrax
    import numpy as np
    import time
    
    # Convert input lists to JAX arrays
    train_x_orig = jnp.array(train_x_list)
    train_y_orig = jnp.array(train_y_list)
    test_x_orig = jnp.array(test_x_list)
    test_y_orig = jnp.array(test_y_list)
    n_train = len(train_x_orig)
    n_test = len(test_x_orig)
    
    
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
    print(f"Training data: {n_train} samples, Test data: {n_test} samples")
    
    # Set random seeds
    key = random.PRNGKey(42)
    np.random.seed(42)
    
    # Architecture: (256, 128, 128, 64) matching infinite data
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
    
    def generate_training_data_from_process(n_samples, key):
        """
        Generate training data from the true generative model (for validation/scatter plots).
        
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
        Generate flow training batch from finite data.
        
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
    
    def diffrax_integrate_single(params, x_val, z0):
        """
        Integrate a single ODE using diffrax from t=0 to t=1.
        
        Args:
            params: Model parameters
            x_val: Single x value (scalar)
            z0: Initial z value (scalar)
        
        Returns:
            Final z value after integration (scalar)
        """
        def vector_field(t, z, args):
            # Unpack args
            params_arg, x_arg = args
            # Create feature array: [x, t, z]
            features = jnp.array([x_arg, t, z])
            return mlp_forward(params_arg, features)
        
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()  # Tsitouras 5/4 method (similar to RK45)
        
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=z0,
            args=(params, x_val),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=1000,
        )
        return solution.ys[-1]
    
    def diffrax_integrate_batch(params, x_batch, z0_batch):
        """
        Integrate ODEs for a batch of initial conditions using diffrax.
        
        Integrates from t=0 to t=1.0 using adaptive step size control.
        
        Args:
            params: Model parameters
            x_batch: Batch of x values (shape: [batch_size])
            z0_batch: Batch of initial z values (shape: [batch_size])
        
        Returns:
            Final z values after integration (shape: [batch_size])
        """
        # vmap over batch dimension
        return vmap(lambda x, z0: diffrax_integrate_single(params, x, z0))(x_batch, z0_batch)
    
    # JIT compile the integration for speed
    diffrax_integrate_batch_jit = jit(diffrax_integrate_batch)
    
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
    
    # Compute standardization stats from training data
    print("Computing standardization stats from training data...")
    x_mean, x_std, y_mean, y_std = standardize_data(train_x_orig, train_y_orig)
    
    # Scale training data and test data
    train_x_scaled, train_y_scaled = transform_data(train_x_orig, train_y_orig, x_mean, x_std, y_mean, y_std)
    test_x_scaled, _ = transform_data(test_x_orig, test_y_orig, x_mean, x_std, y_mean, y_std)
    
    print(f"Standardization stats: x_mean={float(x_mean):.4f}, x_std={float(x_std):.4f}, y_mean={float(y_mean):.4f}, y_std={float(y_std):.4f}")
    
    # Calculate n_t_per_sample to get similar batch size as infinite data
    # Infinite data uses n_train_per_step=90000 with n_t_per_sample=3, giving 270000 flow samples
    # With 900 training points, we need n_t_per_sample = 270000 / 900 = 300 to match
    # This ensures comparable batch sizes between reference (finite) and infinite data training
    n_t_per_sample = 300
    print(f"Using n_t_per_sample={n_t_per_sample} to match infinite data batch size (900 * 300 = 270,000 flow samples)")
    
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
    
    # Number of samples per held-out point for energy score (Task 4: use 90 samples)
    N_ENERGY_SAMPLES = 90
    
    while time.time() < end_time:
        # Check if we've passed the halfway point and need to halve the learning rate
        if not learning_rate_halved and time.time() >= halfway_time:
            new_learning_rate = learning_rate / 2
            print(f"\n{'='*50}")
            print(f"Halfway point reached! Halving learning rate from {learning_rate} to {new_learning_rate}")
            print(f"{'='*50}\n")
            
            # Create new optimizer with halved learning rate
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=new_learning_rate)
            )
            # Reinitialize optimizer state with current params
            # Note: This resets momentum, which is intentional when making a significant LR change
            opt_state = optimizer.init(params)
            learning_rate_halved = True
        
        # Generate flow batch from finite training data (with fresh t and eps each step)
        key, batch_key = random.split(key)
        features, targets = generate_flow_batch(train_x_scaled, train_y_scaled, n_t_per_sample, batch_key)
        
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
            
            # Calculate energy score every 50 steps using the FIXED 100 test points
            # Task 4: Use 90 samples per held-out point instead of 2
            if step % 50 == 0:
                # Use the fixed test data (test_x_orig, test_y_orig) for energy score
                # Generate N_ENERGY_SAMPLES (90) samples per test point
                key, sample_key = random.split(key)
                z0_keys = random.split(sample_key, N_ENERGY_SAMPLES * n_test)  # 100 x values * 90 samples each
                
                # Create batches: repeat each x N_ENERGY_SAMPLES times
                test_x_expanded = jnp.repeat(test_x_scaled, N_ENERGY_SAMPLES)
                z0_samples = jnp.array([random.normal(k, ()) for k in z0_keys])
                
                # Batch integrate all samples at once (fully GPU-accelerated)
                test_samples_scaled = diffrax_integrate_batch_jit(params, test_x_expanded, z0_samples)
                
                # Reshape to (n_test, N_ENERGY_SAMPLES)
                test_samples_scaled = test_samples_scaled.reshape(n_test, N_ENERGY_SAMPLES)
                
                # Transform back to original scale
                test_samples_orig = inverse_transform_y(test_samples_scaled, y_mean, y_std)
                
                # Calculate 90-sample Monte Carlo energy score for each of 100 held-out points
                # Energy Score = E[|Y - X_j|] - 0.5 * E[|X_j - X_j'|]
                # where Y is true, X_j are samples (j=1..90)
                
                # Vectorized Term 1: Average |Y_i - X_ij| for each i, then average over all i
                # test_y_orig has shape (n_test,), test_samples_orig has shape (n_test, N_ENERGY_SAMPLES)
                term1 = jnp.mean(jnp.abs(test_y_orig[:, None] - test_samples_orig))
                
                # Vectorized Term 2: Average |X_ij - X_ij'| for all pairs j != j', then average over all i
                # For each test point, compute pairwise distances between its samples
                # samples_diff[i, j, jp] = samples_i[j] - samples_i[jp]
                samples_diff = test_samples_orig[:, :, None] - test_samples_orig[:, None, :]  # (n_test, N, N)
                samples_abs_diff = jnp.abs(samples_diff)
                # Extract upper triangle (j < jp) and compute mean
                triu_mask = jnp.triu(jnp.ones((N_ENERGY_SAMPLES, N_ENERGY_SAMPLES)), k=1)
                # For each test point, sum upper triangle and divide by number of pairs
                n_pairs = N_ENERGY_SAMPLES * (N_ENERGY_SAMPLES - 1) // 2
                term2_per_test = jnp.sum(samples_abs_diff * triu_mask, axis=(1, 2)) / n_pairs
                term2 = jnp.mean(term2_per_test)
                
                test_energy = float(term1 - 0.5 * term2)
                energy_scores.append(test_energy)
                energy_steps.append(step)
                energy_times.append(elapsed)
                
                print(f"         Energy Score (90 samples per 100 test points): {test_energy:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Generate final samples for 1000 points using JAX-accelerated sampling
    print("\nGenerating 1000 samples for scatter plot...")
    key, sample_key = random.split(key)
    scatter_x, scatter_y = generate_training_data_from_process(1000, sample_key)
    scatter_x_scaled, _ = transform_data(scatter_x, scatter_y, x_mean, x_std, y_mean, y_std)
    
    # Generate random initial conditions for all samples at once
    key, z0_key = random.split(key)
    z0_samples = random.normal(z0_key, (1000,))
    
    # Batch integrate all 1000 samples at once (fully GPU-accelerated)
    print("  Running batch ODE integration on GPU...")
    scatter_samples_scaled = diffrax_integrate_batch_jit(params, scatter_x_scaled, z0_samples)
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
def main(duration_minutes: float = 2):
    """
    Main entrypoint for running training on Modal.
    
    Args:
        duration_minutes: How long to train (in minutes)
    """
    import csv
    import numpy as np
    from pathlib import Path
    
    print(f"Starting reference model training on Modal with T4 GPU for {duration_minutes} minutes...")
    
    # Load training data from dataset1
    script_dir = Path(__file__).parent
    dataset_dir = script_dir.parent.parent / "dataset1" / "data"
    
    print(f"Loading training data from {dataset_dir / 'train.npy'}...")
    train_data = np.load(dataset_dir / "train.npy")
    train_x = train_data[:, 0].astype(np.float32).tolist()
    train_y = train_data[:, 1].astype(np.float32).tolist()
    print(f"Loaded {len(train_x)} training samples")
    
    # Load held-out test data from dataset1
    print(f"Loading test data from {dataset_dir / 'test_x.npy'} and {dataset_dir / 'test_y.npy'}...")
    test_x = np.load(dataset_dir / "test_x.npy").astype(np.float32).tolist()
    test_y = np.load(dataset_dir / "test_y.npy").astype(np.float32).tolist()
    print(f"Loaded {len(test_x)} test samples")
    
    # Run training on Modal
    result = train_model.remote(
        train_x_list=train_x,
        train_y_list=train_y,
        test_x_list=test_x,
        test_y_list=test_y,
        duration_minutes=duration_minutes
    )
    
    # Create output directory
    output_dir = script_dir / "modal_outputs"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving results to {output_dir}...")
    
    # Save CSVs using standard library csv module
    # Training loss CSV - simplified to just train_loss
    with open(output_dir / "reference_training_loss.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'time_seconds'])
        for step, loss, time_val in zip(
            result['steps'], result['train_losses'], result['times']
        ):
            writer.writerow([step, loss, time_val])
    
    # Energy score CSV (only every 50 steps, using 90-sample Monte Carlo)
    if len(result['energy_scores']) > 0:
        with open(output_dir / "reference_energy_score.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'energy_score', 'time_seconds'])
            for step, score, time_val in zip(result['energy_steps'], result['energy_scores'], result['energy_times']):
                writer.writerow([step, score, time_val])
    
    # Scatter samples CSV
    with open(output_dir / "reference_scatter_samples.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y_true', 'y_sampled'])
        for x, y_true, y_sampled in zip(result['scatter_x'], result['scatter_y'], result['scatter_samples']):
            writer.writerow([x, y_true, y_sampled])
    
    print("\nAll outputs saved successfully!")
    print(f"  - reference_training_loss.csv")
    if len(result['energy_scores']) > 0:
        print(f"  - reference_energy_score.csv (90-sample Monte Carlo on 100 test points)")
    print(f"  - reference_scatter_samples.csv")
    print(f"\nTotal training time: {result['total_time']:.2f} seconds ({result['total_time']/60:.2f} minutes)")
    print(f"\nNote: CSV files saved. You can visualize them with pandas/matplotlib:")
    print(f"  import pandas as pd; import matplotlib.pyplot as plt")
    print(f"  df = pd.read_csv('{output_dir / 'reference_training_loss.csv'}')")
    print(f"  plt.plot(df['time_seconds'], df['train_loss']); plt.show()")
