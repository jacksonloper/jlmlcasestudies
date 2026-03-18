"""
Train reference solution for Case Study 5: Likelihood Estimation using JAX on Modal.com with T4 GPU.

This script implements rectified flow matching for density estimation. The model learns
a velocity field v(x, t, z_t) that transforms noise z_0 ~ N(0,1) to conditional samples
z_1 ~ p(y|x1, x2).

To compute log-likelihoods, we use the continuous normalizing flow (CNF) approach:
evolve the augmented state (z, log_likelihood) backwards from t=1 (data) to t=0 (noise),
using the instantaneous change of variables formula:

    dz/dt = v(x, t, z)
    d(log p)/dt = -div_z v(x, t, z)

Since z is 1-dimensional, div_z v = dv/dz, computed via exact autodiff (not trace estimation).

We integrate from t=1 (data point y) backwards to t=0 (noise), accumulating
the log-density change. The final log-likelihood is:

    log p(y|x) = log p_0(z_0) - integral_0^1 div_z v(x, t, z_t) dt

where p_0 = N(0, 1) is the base distribution.

Uses Dormand-Prince (dopri5) adaptive solver for accuracy.

Outputs:
- reference_training_loss.csv: Training loss over time
- reference_loglik_mse.csv: MSE of log-likelihood estimates over training
- reference_loglik_scatter.csv: True vs estimated log-likelihoods for scatter plot
- reference_generated_samples.csv: Samples generated from the trained flow model
"""

import modal

# Create Modal app
app = modal.App("case5-reference-jax")

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
def train_model(train_x_list, train_y_list, test_x_list, test_y_list,
                test_true_loglik_list, duration_minutes=5, learning_rate=0.0001,
                batch_size=4096, weight_decay=1e-4, infinite_data=False):
    """
    Train rectified flow model and compute log-likelihoods using augmented ODE.

    Args:
        train_x_list: Training x values as list of [x1, x2] pairs
        train_y_list: Training y values as list
        test_x_list: Test x values as list of [x1, x2] pairs
        test_y_list: Test y values as list
        test_true_loglik_list: True log-likelihoods for test data
        duration_minutes: How long to train (in minutes)
        learning_rate: Learning rate for AdamW optimizer
        batch_size: Minibatch size for training
        weight_decay: Weight decay for AdamW optimizer
        infinite_data: If True, generate fresh training data each step (for debugging)

    Returns:
        Dictionary with training history and log-likelihood estimates
    """
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap, jacfwd
    import optax
    import diffrax
    import numpy as np
    import time

    # Convert input lists to JAX arrays
    train_x = jnp.array(train_x_list)  # (n_train, 2)
    train_y = jnp.array(train_y_list)  # (n_train,)
    test_x = jnp.array(test_x_list)    # (n_test, 2)
    test_y = jnp.array(test_y_list)    # (n_test,)
    test_true_loglik = jnp.array(test_true_loglik_list)  # (n_test,)
    n_train = len(train_y)
    n_test = len(test_y)

    # Verify GPU is available
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    backend = jax.default_backend()
    print(f"JAX backend: {backend}")

    if backend != "gpu" and jax.devices()[0].platform != "gpu":
        raise RuntimeError(
            f"Expected GPU backend but got '{backend}'. "
            f"Device platform: {jax.devices()[0].platform}. "
            "This script requires GPU acceleration."
        )

    print(f"✓ GPU backend confirmed")
    print(f"Training for {duration_minutes} minutes with lr={learning_rate}, batch_size={batch_size}, weight_decay={weight_decay}")
    print(f"Training data: {n_train} samples, Test data: {n_test} samples")
    if infinite_data:
        print("*** INFINITE DATA MODE: generating fresh training data each step ***")

    # Set random seeds
    key = random.PRNGKey(42)
    np.random.seed(42)

    # --- Data generation function (for infinite data mode) ---
    def generate_data_from_process(n_samples, key):
        """
        Generate fresh (x, y) pairs from the true generative model:
        X1, X2 iid ~ 0.5*N(-2, 1) + 0.5*N(2, 1)
        Y | X ~ 0.5*N(X1, 1) + 0.5*N(X2, 1)
        """
        k1, k2, k3, k4, k5, k6, k7 = random.split(key, 7)
        # X1 ~ mixture of N(-2, 1) and N(2, 1)
        x1_selector = random.uniform(k1, (n_samples,)) < 0.5
        x1_comp1 = random.normal(k2, (n_samples,)) + (-2.0)
        x1_comp2 = random.normal(k3, (n_samples,)) + 2.0
        x1 = jnp.where(x1_selector, x1_comp1, x1_comp2)
        # X2 ~ mixture of N(-2, 1) and N(2, 1)
        x2_selector = random.uniform(k4, (n_samples,)) < 0.5
        x2_comp1 = random.normal(k5, (n_samples,)) + (-2.0)
        x2_comp2 = random.normal(k6, (n_samples,)) + 2.0
        x2 = jnp.where(x2_selector, x2_comp1, x2_comp2)
        # Y | X ~ mixture: flip coin, then sample from component
        k_sel, k_y = random.split(k7)
        selector = random.uniform(k_sel, (n_samples,)) < 0.5
        y1 = random.normal(k_y, (n_samples,)) + x1  # N(x1, 1)
        k_y2 = random.fold_in(k_y, 1)
        y2 = random.normal(k_y2, (n_samples,)) + x2  # N(x2, 1)
        y = jnp.where(selector, y1, y2)
        x = jnp.column_stack([x1, x2])
        return x, y

    # Architecture: (256, 128, 128, 64)
    # Input: x1, x2, t, zt (4 features)
    # Output: velocity (1 scalar)
    hidden_layers = [256, 128, 128, 64]
    input_dim = 4  # x1, x2, t, zt
    output_dim = 1  # velocity field

    def init_network_params(layer_sizes, key):
        """Initialize MLP parameters with Xavier initialization."""
        params = []
        keys = random.split(key, len(layer_sizes))
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            k1, k2 = random.split(keys[i])
            w = random.normal(k1, (n_in, n_out)) * jnp.sqrt(2.0 / (n_in + n_out))
            b = jnp.zeros(n_out)
            params.append((w, b))
        return params

    def mlp_forward(params, x):
        """Forward pass through MLP with ReLU activations."""
        for i, (w, b) in enumerate(params[:-1]):
            x = jnp.dot(x, w) + b
            x = jax.nn.relu(x)
        w, b = params[-1]
        x = jnp.dot(x, w) + b
        return x.squeeze()  # Return scalar

    def standardize_data(x, y):
        """Compute standardization statistics."""
        x_mean = jnp.mean(x, axis=0)  # (2,) for 2D input
        x_std = jnp.std(x, axis=0)
        y_mean = jnp.mean(y)
        y_std = jnp.std(y)
        return x_mean, x_std, y_mean, y_std

    def transform_x(x, x_mean, x_std):
        """Standardize x values."""
        return (x - x_mean) / (x_std + 1e-8)

    def transform_y(y, y_mean, y_std):
        """Standardize y values."""
        return (y - y_mean) / (y_std + 1e-8)

    def inverse_transform_y(y_scaled, y_mean, y_std):
        """Inverse transform y."""
        return y_scaled * y_std + y_mean

    def generate_flow_batch(x_data, y_data, n_t_per_sample, key):
        """
        Generate flow training batch from finite data.

        For each sample, draws n_t_per_sample independent random t values.
        Uses linear interpolation: z_t = t*y + (1-t)*eps
        Target velocity: v = y - eps
        """
        n_samples = len(y_data)
        n_total = n_samples * n_t_per_sample

        # Replicate each sample n_t_per_sample times
        x_expanded = jnp.repeat(x_data, n_t_per_sample, axis=0)  # (n_total, 2)
        y_expanded = jnp.repeat(y_data, n_t_per_sample)  # (n_total,)

        # Generate t values from Beta(2, 2)
        k_eps, k_t = random.split(key)
        t_values = random.beta(k_t, 2.0, 2.0, shape=(n_total,))

        # Generate random noise
        eps_values = random.normal(k_eps, (n_total,))

        # Compute z_t = y*t + (1-t)*eps
        zt_values = y_expanded * t_values + (1 - t_values) * eps_values

        # Target: y - eps
        targets = y_expanded - eps_values

        # Create features (x1, x2, t, zt)
        features = jnp.column_stack([x_expanded, t_values[:, None], zt_values[:, None]])

        return features, targets

    @jit
    def loss_fn(params, features, targets):
        """MSE loss."""
        predictions = vmap(lambda x: mlp_forward(params, x))(features)
        return jnp.mean((predictions - targets) ** 2)

    # --- Log-likelihood computation using augmented ODE ---

    def velocity_fn(params, x1_val, x2_val, t, z):
        """Compute velocity v(x, t, z) from the network."""
        features = jnp.array([x1_val, x2_val, t, z])
        return mlp_forward(params, features)

    def div_velocity_fn(params, x1_val, x2_val, t, z):
        """Compute dv/dz (exact divergence in 1D) using autodiff."""
        return grad(lambda z_: velocity_fn(params, x1_val, x2_val, t, z_))(z)

    def compute_loglik_single(params, x1_val, x2_val, y_val, y_mean, y_std):
        """
        Compute log p(y|x) for a single data point using the augmented ODE.

        Integrate backwards from t=1 (data) to t=0 (noise):
            dz/dt = v(x, t, z)
            d(log_lik_change)/dt = -div_z v(x, t, z)

        Then: log p(y|x) = log p_0(z_0) - log_lik_change - log(y_std)
        The -log(y_std) term accounts for the change of variables from
        standardized to original scale.
        """
        y_scaled = (y_val - y_mean) / (y_std + 1e-8)

        def augmented_dynamics(t, state, args):
            z = state[0]
            params_arg, x1_arg, x2_arg = args
            v = velocity_fn(params_arg, x1_arg, x2_arg, t, z)
            div_v = div_velocity_fn(params_arg, x1_arg, x2_arg, t, z)
            return jnp.array([v, -div_v])

        term = diffrax.ODETerm(augmented_dynamics)
        solver = diffrax.Dopri5()

        # Integrate from t=1 (data) to t=0 (noise) by going backwards
        # We reverse: integrate from t=0 to t=1 with reversed dynamics
        # Or equivalently: set t0=1, t1=0 with dt0=-0.01
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=1.0,
            t1=0.0,
            dt0=-0.01,
            y0=jnp.array([y_scaled, 0.0]),  # [z_1=y_scaled, accumulated_loglik=0]
            args=(params, x1_val, x2_val),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=4000,
        )

        z0 = solution.ys[0, -1]       # Final z value (should be ~N(0,1))
        loglik_change = solution.ys[1, -1]  # Accumulated log-likelihood change

        # CNF change of variables: log p(y_scaled) = log p_0(z_0) - ∫_0^1 ∇·v dt
        # The augmented ODE dℓ/dt = -∇·v integrated from t=1→0 gives:
        #   ℓ(0) = ∫_0^1 ∇·v dt  (positive)
        # So: log p(y_scaled) = log p_0(z_0) - ℓ(0)
        # The -log(y_std) accounts for change of variables from standardized to original scale.
        # p_0(z) = N(z; 0, 1) => log p_0(z) = -0.5*z^2 - 0.5*log(2*pi)
        log_p0 = -0.5 * z0**2 - 0.5 * jnp.log(2.0 * jnp.pi)
        log_py = log_p0 - loglik_change - jnp.log(y_std + 1e-8)

        return log_py

    # Initialize model
    print("Initializing model...")
    layer_sizes = [input_dim] + hidden_layers + [output_dim]
    key, init_key = random.split(key)
    params = init_network_params(layer_sizes, init_key)

    # Initialize optimizer with weight decay for regularization
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )
    opt_state = optimizer.init(params)

    @jit
    def update_step(params, opt_state, features, targets):
        """Single optimization step."""
        loss = loss_fn(params, features, targets)
        grads = grad(loss_fn)(params, features, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Compute standardization stats
    if infinite_data:
        # For infinite data mode, compute stats from a large sample
        print("Computing standardization stats from large sample (50,000 points)...")
        key, stats_key = random.split(key)
        stats_x, stats_y = generate_data_from_process(50000, stats_key)
        x_mean, x_std, y_mean, y_std = standardize_data(stats_x, stats_y)
        del stats_x, stats_y  # Free memory
    else:
        print("Computing standardization stats from training data...")
        x_mean, x_std, y_mean, y_std = standardize_data(train_x, train_y)
    print(f"Stats: x_mean={x_mean}, x_std={x_std}, y_mean={float(y_mean):.4f}, y_std={float(y_std):.4f}")

    # Scale data
    train_x_scaled = transform_x(train_x, x_mean, x_std)
    train_y_scaled = transform_y(train_y, y_mean, y_std)
    test_x_scaled = transform_x(test_x, x_mean, x_std)

    # Number of t values per sample
    if infinite_data:
        # With infinite data, use 5000 fresh samples * 54 t-values = 270,000 flow samples per step
        n_samples_per_step = 5000
        n_t_per_sample = 54
        print(f"Infinite data: generating {n_samples_per_step} fresh samples * {n_t_per_sample} t-values = {n_samples_per_step * n_t_per_sample} flow samples per step")
    else:
        n_t_per_sample = 54
        print(f"Using n_t_per_sample={n_t_per_sample} ({n_train} * {n_t_per_sample} = {n_train * n_t_per_sample} flow samples per step)")

    # Pre-generate test flow batch for test MSE evaluation
    test_y_scaled = transform_y(test_y, y_mean, y_std)
    test_flow_key = random.PRNGKey(9999)
    test_flow_features, test_flow_targets = generate_flow_batch(
        test_x_scaled, test_y_scaled, n_t_per_sample, test_flow_key
    )
    print(f"Test flow batch size: {len(test_flow_features)} samples")

    # --- Compute log-likelihoods function (uses current params via closure) ---
    def calc_loglik_metrics(current_params, key):
        """Compute log-likelihood metrics on test set: MSE and mean estimated log-lik."""
        # We need to rebuild the vmap with current params
        def single_loglik(x1, x2, y):
            return compute_loglik_single(current_params, x1, x2, y, y_mean, y_std)

        batch_loglik = jit(vmap(single_loglik, in_axes=(0, 0, 0)))

        estimated_loglik = batch_loglik(
            test_x_scaled[:, 0], test_x_scaled[:, 1], test_y
        )

        mse = jnp.mean((estimated_loglik - test_true_loglik) ** 2)
        mean_loglik = jnp.mean(estimated_loglik)
        return float(mse), float(mean_loglik), np.array(estimated_loglik), key

    # Training loop
    print(f"\nStarting training for {duration_minutes} minutes...")
    print(f"{'Step':>8}  {'Train Loss':>12}  {'Test MSE':>12}  {'Time':>8}")
    print("-" * 50)

    start_time = time.time()
    end_time = start_time + duration_minutes * 60

    step = 0
    train_losses = []
    test_mses = []
    steps_recorded = []
    times_recorded = []
    loglik_mse_values = []
    loglik_mean_values = []
    loglik_mse_steps = []
    loglik_mse_times = []

    # Track the best model based on log-likelihood MSE
    import copy
    best_loglik_mse = float('inf')
    best_params = None
    best_step = 0

    while time.time() < end_time:
        # Generate flow batch
        key, batch_key = random.split(key)

        if infinite_data:
            # Generate fresh training data each step
            key, data_key = random.split(key)
            fresh_x, fresh_y = generate_data_from_process(n_samples_per_step, data_key)
            fresh_x_scaled = transform_x(fresh_x, x_mean, x_std)
            fresh_y_scaled = transform_y(fresh_y, y_mean, y_std)
            features, targets = generate_flow_batch(
                fresh_x_scaled, fresh_y_scaled, n_t_per_sample, batch_key
            )
        else:
            features, targets = generate_flow_batch(
                train_x_scaled, train_y_scaled, n_t_per_sample, batch_key
            )

        # Shuffle
        key, shuffle_key = random.split(key)
        perm = random.permutation(shuffle_key, len(features))
        features = features[perm]
        targets = targets[perm]

        # Minibatched training
        n_batches = len(features) // batch_size
        loss_sum = 0.0

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_features = features[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]

            params, opt_state, batch_loss = update_step(
                params, opt_state, batch_features, batch_targets
            )
            loss_sum += batch_loss

        train_loss = (loss_sum / n_batches).item()
        step += 1

        # Record every 10 steps
        if step % 10 == 0:
            elapsed = time.time() - start_time
            test_mse = float(loss_fn(params, test_flow_features, test_flow_targets))

            train_losses.append(train_loss)
            test_mses.append(test_mse)
            steps_recorded.append(step)
            times_recorded.append(elapsed)

            print(f"{step:8d}  {train_loss:12.6f}  {test_mse:12.6f}  {elapsed:6.1f}s")

            # Compute log-likelihood metrics every 100 steps
            if step % 100 == 0:
                try:
                    lmse, lmean, _, key = calc_loglik_metrics(params, key)
                    loglik_mse_values.append(lmse)
                    loglik_mean_values.append(lmean)
                    loglik_mse_steps.append(step)
                    loglik_mse_times.append(elapsed)

                    # Track best model
                    if lmse < best_loglik_mse:
                        best_loglik_mse = lmse
                        best_params = jax.tree.map(lambda x: x.copy(), params)
                        best_step = step
                        print(f"         Log-lik MSE: {lmse:.4f}, mean est. log-lik: {lmean:.4f} *** new best ***")
                    else:
                        print(f"         Log-lik MSE: {lmse:.4f}, mean est. log-lik: {lmean:.4f} (best: {best_loglik_mse:.4f} at step {best_step})")
                except Exception as e:
                    print(f"         Log-lik metrics computation failed: {e}")

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.2f}s ({total_time/60:.2f} min)")

    # Compute true mean log-lik for reference
    true_mean_loglik = float(jnp.mean(test_true_loglik))
    print(f"True mean log-likelihood on test data: {true_mean_loglik:.4f}")

    # Use best model for final log-likelihood computation
    eval_params = best_params if best_params is not None else params
    print(f"\nComputing final log-likelihood estimates using best model (step {best_step}, MSE={best_loglik_mse:.4f})...")
    try:
        final_mse, final_mean_loglik, estimated_logliks, key = calc_loglik_metrics(eval_params, key)
        print(f"Final log-likelihood MSE: {final_mse:.4f}")
        print(f"Estimated log-liks: mean={np.mean(estimated_logliks):.4f}, std={np.std(estimated_logliks):.4f}")
        print(f"True log-liks:      mean={float(jnp.mean(test_true_loglik)):.4f}, std={float(jnp.std(test_true_loglik)):.4f}")
    except Exception as e:
        print(f"Final log-likelihood computation failed: {e}")
        estimated_logliks = np.full(n_test, np.nan)
        final_mse = float('nan')

    # --- Generate samples from the trained flow model ---
    def generate_sample_single(params, x1_scaled, x2_scaled, z0):
        """
        Generate a sample by integrating the ODE forward from t=0 (noise) to t=1 (data).
        z0 ~ N(0,1) in standardized space, returns y in original space.
        """
        def dynamics(t, z, args):
            params_arg, x1_arg, x2_arg = args
            return velocity_fn(params_arg, x1_arg, x2_arg, t, z)

        term = diffrax.ODETerm(dynamics)
        solver = diffrax.Dopri5()

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=z0,
            args=(params, x1_scaled, x2_scaled),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=4000,
        )

        y_scaled = solution.ys[-1]
        return inverse_transform_y(y_scaled, y_mean, y_std)

    n_gen_samples = 500
    print(f"\nGenerating {n_gen_samples} samples from the trained flow model...")
    try:
        key, gen_key = random.split(key)
        # Generate fresh x values from the prior
        gen_x, _ = generate_data_from_process(n_gen_samples, gen_key)
        gen_x_scaled = transform_x(gen_x, x_mean, x_std)

        # Sample noise z0 ~ N(0,1)
        key, z0_key = random.split(key)
        z0_samples = random.normal(z0_key, (n_gen_samples,))

        # Generate y values by integrating ODE forward
        batch_generate = jit(vmap(
            lambda x1s, x2s, z0: generate_sample_single(eval_params, x1s, x2s, z0),
            in_axes=(0, 0, 0)
        ))
        gen_y = batch_generate(gen_x_scaled[:, 0], gen_x_scaled[:, 1], z0_samples)
        gen_y = np.array(gen_y)
        gen_x_np = np.array(gen_x)

        print(f"Generated samples: y mean={np.mean(gen_y):.4f}, std={np.std(gen_y):.4f}")
        print(f"Generated samples: x1 mean={np.mean(gen_x_np[:, 0]):.4f}, x2 mean={np.mean(gen_x_np[:, 1]):.4f}")
    except Exception as e:
        print(f"Sample generation failed: {e}")
        gen_x_np = np.zeros((0, 2))
        gen_y = np.zeros(0)

    return {
        'steps': steps_recorded,
        'train_losses': train_losses,
        'test_mses': test_mses,
        'times': times_recorded,
        'loglik_mse_values': loglik_mse_values,
        'loglik_mean_values': loglik_mean_values,
        'loglik_mse_steps': loglik_mse_steps,
        'loglik_mse_times': loglik_mse_times,
        'estimated_logliks': estimated_logliks.tolist(),
        'true_logliks': np.array(test_true_loglik).tolist(),
        'test_x1': np.array(test_x[:, 0]).tolist(),
        'test_x2': np.array(test_x[:, 1]).tolist(),
        'test_y': np.array(test_y).tolist(),
        'total_time': total_time,
        'final_mse': final_mse,
        'true_mean_loglik': true_mean_loglik,
        'generated_x1': gen_x_np[:, 0].tolist() if len(gen_x_np) > 0 else [],
        'generated_x2': gen_x_np[:, 1].tolist() if len(gen_x_np) > 0 else [],
        'generated_y': gen_y.tolist() if len(gen_y) > 0 else [],
    }


@app.local_entrypoint()
def main(duration_minutes: float = 5, weight_decay: float = 1e-4, infinite_data: bool = False):
    """
    Main entrypoint for running training on Modal.

    Args:
        duration_minutes: How long to train (in minutes)
        weight_decay: Weight decay for AdamW optimizer
        infinite_data: If True, generate fresh training data each step (for debugging)
    """
    import csv
    import numpy as np
    from pathlib import Path

    mode_str = " [INFINITE DATA]" if infinite_data else ""
    print(f"Starting Case 5 reference model training on Modal with T4 GPU for {duration_minutes} minutes (weight_decay={weight_decay}){mode_str}...")

    # Load data
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    print(f"Loading data from {data_dir}...")
    train_x = np.load(data_dir / "train_x.npy").astype(np.float32).tolist()
    train_y = np.load(data_dir / "train_y.npy").astype(np.float32).tolist()
    test_x = np.load(data_dir / "test_x.npy").astype(np.float32).tolist()
    test_y = np.load(data_dir / "test_y.npy").astype(np.float32).tolist()
    test_true_loglik = np.load(data_dir / "test_true_loglik.npy").astype(np.float32).tolist()

    print(f"Loaded {len(train_y)} training samples, {len(test_y)} test samples")

    # Run training on Modal
    result = train_model.remote(
        train_x_list=train_x,
        train_y_list=train_y,
        test_x_list=test_x,
        test_y_list=test_y,
        test_true_loglik_list=test_true_loglik,
        duration_minutes=duration_minutes,
        weight_decay=weight_decay,
        infinite_data=infinite_data
    )

    # Create output directory
    output_dir = Path(__file__).parent / "modal_outputs"
    output_dir.mkdir(exist_ok=True)

    print(f"\nSaving results to {output_dir}...")

    # Training loss CSV
    with open(output_dir / "reference_training_loss.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'test_mse', 'time_seconds'])
        for step, loss, test_mse, time_val in zip(
            result['steps'], result['train_losses'], result['test_mses'], result['times']
        ):
            writer.writerow([step, loss, test_mse, time_val])

    # Log-likelihood MSE CSV (with mean estimated log-likelihood)
    if len(result['loglik_mse_values']) > 0:
        with open(output_dir / "reference_loglik_mse.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loglik_mse', 'loglik_mean', 'time_seconds'])
            for step, mse, mean_ll, time_val in zip(
                result['loglik_mse_steps'], result['loglik_mse_values'],
                result['loglik_mean_values'], result['loglik_mse_times']
            ):
                writer.writerow([step, mse, mean_ll, time_val])

    # Log-likelihood scatter CSV (true vs estimated)
    with open(output_dir / "reference_loglik_scatter.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['test_x1', 'test_x2', 'test_y', 'true_loglik', 'estimated_loglik'])
        for x1, x2, y, true_ll, est_ll in zip(
            result['test_x1'], result['test_x2'], result['test_y'],
            result['true_logliks'], result['estimated_logliks']
        ):
            writer.writerow([x1, x2, y, true_ll, est_ll])

    # Generated samples CSV
    if len(result.get('generated_x1', [])) > 0:
        with open(output_dir / "reference_generated_samples.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x1', 'x2', 'generated_y'])
            for x1, x2, y in zip(
                result['generated_x1'], result['generated_x2'], result['generated_y']
            ):
                writer.writerow([x1, x2, y])

    print("\nAll outputs saved successfully!")
    print(f"  - reference_training_loss.csv")
    if len(result['loglik_mse_values']) > 0:
        print(f"  - reference_loglik_mse.csv (includes mean estimated log-lik)")
    print(f"  - reference_loglik_scatter.csv")
    if len(result.get('generated_x1', [])) > 0:
        print(f"  - reference_generated_samples.csv ({len(result['generated_x1'])} samples)")
    print(f"\nFinal log-likelihood MSE: {result['final_mse']:.4f}")
    print(f"True mean log-likelihood: {result['true_mean_loglik']:.4f}")
    print(f"Total training time: {result['total_time']:.2f}s ({result['total_time']/60:.2f} min)")
