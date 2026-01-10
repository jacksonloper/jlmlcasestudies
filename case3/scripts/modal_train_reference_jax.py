"""
Train reference solution for Case Study 3: Modular Arithmetic using JAX on Modal.com with T4 GPU.

This script trains neural networks to learn modular addition mod 97.
Architecture: 194 (input) -> hidden -> hidden -> 97 (output) with ReLU activations

Two training runs are performed:
1. Adam (no weight decay) - Shows memorization without generalization
2. AdamW (with weight decay) - Shows grokking phenomenon (delayed generalization)

Outputs:
- reference_training_loss.csv: Training history WITHOUT weight decay (memorization)
- reference_training_loss_wd.csv: Training history WITH weight decay (grokking)
"""

import modal

# Create Modal app
app = modal.App("case3-reference-jax")

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
    timeout=90 * 60,  # 90 minute timeout for extended training (both runs)
)
def train_model(train_x_list, train_y_list, test_x_list, test_y_list, 
                hidden_size=128, n_epochs=50000, learning_rate=0.001, batch_size=512,
                log_interval_epochs=10, weight_decay=0.0, run_name="default"):
    """
    Train a neural network for modular arithmetic using JAX.
    
    Args:
        train_x_list: Training inputs as list of lists (one-hot encoded)
        train_y_list: Training labels as list
        test_x_list: Test inputs as list of lists (one-hot encoded)
        test_y_list: Test labels as list
        hidden_size: Size of hidden layers
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        batch_size: Minibatch size
        log_interval_epochs: Log metrics every N epochs (for CSV storage)
        weight_decay: Weight decay for AdamW (0.0 = no weight decay)
        run_name: Name for this training run
    
    Returns:
        Dictionary with training history
    """
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import optax
    import numpy as np
    import time
    
    # Convert input lists to JAX arrays
    train_x = jnp.array(train_x_list, dtype=jnp.float32)
    train_y = jnp.array(train_y_list, dtype=jnp.int32)
    test_x = jnp.array(test_x_list, dtype=jnp.float32)
    test_y = jnp.array(test_y_list, dtype=jnp.int32)
    
    n_train = len(train_x)
    n_test = len(test_x)
    n_classes = 97
    input_dim = 194  # 97 * 2 one-hot encoding
    
    # Verify GPU is available
    print(f"\n=== {run_name} ===")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    backend = jax.default_backend()
    print(f"JAX backend: {backend}")
    
    if backend != "gpu" and jax.devices()[0].platform != "gpu":
        raise RuntimeError(f"Expected GPU backend but got '{backend}'")
    
    print(f"✓ GPU backend confirmed")
    print(f"Training for {n_epochs} epochs with lr={learning_rate}, batch_size={batch_size}")
    print(f"Weight decay: {weight_decay}")
    print(f"Training data: {n_train} samples, Test data: {n_test} samples")
    print(f"Architecture: {input_dim} -> {hidden_size} -> {hidden_size} -> {n_classes}")
    print(f"Logging every {log_interval_epochs} epochs")
    
    # Set random seed
    key = random.PRNGKey(42)
    
    def init_network_params(key):
        """Initialize two-hidden-layer network with Xavier initialization."""
        keys = random.split(key, 3)
        
        # Layer 1: input_dim -> hidden_size
        w1 = random.normal(keys[0], (input_dim, hidden_size)) * jnp.sqrt(2.0 / (input_dim + hidden_size))
        b1 = jnp.zeros(hidden_size)
        
        # Layer 2: hidden_size -> hidden_size
        w2 = random.normal(keys[1], (hidden_size, hidden_size)) * jnp.sqrt(2.0 / (2 * hidden_size))
        b2 = jnp.zeros(hidden_size)
        
        # Output layer: hidden_size -> n_classes
        w3 = random.normal(keys[2], (hidden_size, n_classes)) * jnp.sqrt(2.0 / (hidden_size + n_classes))
        b3 = jnp.zeros(n_classes)
        
        return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3}
    
    def forward(params, x):
        """Forward pass: input -> hidden -> hidden -> logits with ReLU activations."""
        # Layer 1
        h1 = jnp.dot(x, params['w1']) + params['b1']
        h1 = jax.nn.relu(h1)
        
        # Layer 2
        h2 = jnp.dot(h1, params['w2']) + params['b2']
        h2 = jax.nn.relu(h2)
        
        # Output layer (logits, no activation)
        logits = jnp.dot(h2, params['w3']) + params['b3']
        return logits
    
    def cross_entropy_loss(params, x_batch, y_batch):
        """Compute cross-entropy loss."""
        logits = vmap(lambda x: forward(params, x))(x_batch)
        # Softmax cross-entropy
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Select log prob of correct class
        loss = -jnp.mean(log_probs[jnp.arange(len(y_batch)), y_batch])
        return loss
    
    def compute_accuracy(params, x_data, y_data):
        """Compute classification accuracy."""
        logits = vmap(lambda x: forward(params, x))(x_data)
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(predictions == y_data)
    
    # Initialize model
    print("Initializing model...")
    key, init_key = random.split(key)
    params = init_network_params(init_key)
    
    # Initialize optimizer - Adam or AdamW depending on weight_decay
    if weight_decay > 0:
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        print(f"Using AdamW optimizer with weight_decay={weight_decay}")
    else:
        optimizer = optax.adam(learning_rate=learning_rate)
        print(f"Using Adam optimizer (no weight decay)")
    opt_state = optimizer.init(params)
    
    @jit
    def update_step(params, opt_state, x_batch, y_batch):
        """Single optimization step."""
        loss = cross_entropy_loss(params, x_batch, y_batch)
        grads = grad(cross_entropy_loss)(params, x_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    @jit
    def compute_metrics(params, x_data, y_data):
        """Compute loss and accuracy."""
        logits = vmap(lambda x: forward(params, x))(x_data)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(log_probs[jnp.arange(len(y_data)), y_data])
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == y_data)
        return loss, accuracy
    
    def compute_weight_norm(params):
        """Compute total L2 norm of all weights (excluding biases)."""
        total_norm_sq = 0.0
        for key in ['w1', 'w2', 'w3']:
            total_norm_sq += jnp.sum(params[key] ** 2)
        return jnp.sqrt(total_norm_sq)
    
    # Training loop
    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"{'Epoch':>8} {'Train Loss':>12} {'Test Loss':>12} {'Train Acc':>10} {'Test Acc':>10} {'W Norm':>10} {'Time':>8}")
    print("-" * 80)
    
    start_time = time.time()
    
    epochs_recorded = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    weight_norms = []
    times_recorded = []
    
    # Number of batches per epoch
    n_batches = max(1, n_train // batch_size)
    
    for epoch in range(n_epochs + 1):
        # Shuffle training data each epoch
        key, shuffle_key = random.split(key)
        perm = random.permutation(shuffle_key, n_train)
        train_x_shuffled = train_x[perm]
        train_y_shuffled = train_y[perm]
        
        # Train for one epoch
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            x_batch = train_x_shuffled[start_idx:end_idx]
            y_batch = train_y_shuffled[start_idx:end_idx]
            
            params, opt_state, _ = update_step(params, opt_state, x_batch, y_batch)
        
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Log metrics every log_interval_epochs epochs (for CSV storage)
        # Also always log first (epoch 0) and last epoch
        if epoch == 0 or epoch % log_interval_epochs == 0 or epoch == n_epochs:
            train_loss, train_acc = compute_metrics(params, train_x, train_y)
            test_loss, test_acc = compute_metrics(params, test_x, test_y)
            weight_norm = compute_weight_norm(params)
            
            train_loss = float(train_loss)
            test_loss = float(test_loss)
            train_acc = float(train_acc)
            test_acc = float(test_acc)
            weight_norm = float(weight_norm)
            
            epochs_recorded.append(epoch)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            weight_norms.append(weight_norm)
            times_recorded.append(elapsed)
            
            print(f"{epoch:8d} {train_loss:12.4f} {test_loss:12.4f} {train_acc:10.4f} {test_acc:10.4f} {weight_norm:10.2f} {elapsed:8.1f}s")
            
            # Early stopping if both train and test are near perfect
            if train_acc > 0.999 and test_acc > 0.999:
                print(f"\nEarly stopping: Both train and test accuracy > 99.9%")
                break
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return {
        'epochs': epochs_recorded,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'weight_norms': weight_norms,
        'times': times_recorded,
        'total_time': total_time,
        'run_name': run_name,
    }


@app.local_entrypoint()
def main(hidden_size: int = 128, n_epochs: int = 50000, learning_rate: float = 0.001, 
         batch_size: int = 512, log_interval_epochs: int = 10, weight_decay: float = 0.0,
         output_suffix: str = ""):
    """
    Main entrypoint for running training on Modal.
    
    Args:
        hidden_size: Size of hidden layers
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam
        batch_size: Minibatch size
        log_interval_epochs: Log every N epochs (for CSV storage)
        weight_decay: Weight decay for AdamW (0.0 = no weight decay, try 1.0 for grokking)
        output_suffix: Suffix for output filename (e.g., "_wd" for weight decay run)
    """
    import csv
    import numpy as np
    from pathlib import Path
    
    print(f"Starting Case 3 training on Modal with T4 GPU...")
    print(f"Architecture: 194 -> {hidden_size} -> {hidden_size} -> 97")
    print(f"Epochs: {n_epochs}, LR: {learning_rate}, Batch size: {batch_size}")
    print(f"Weight decay: {weight_decay}")
    print(f"Logging every {log_interval_epochs} epochs")
    
    # Load training and test data
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    print(f"\nLoading data from {data_dir}...")
    train_x = np.load(data_dir / "train_x.npy").astype(np.float32).tolist()
    train_y = np.load(data_dir / "train_y.npy").astype(np.int32).tolist()
    test_x = np.load(data_dir / "test_x.npy").astype(np.float32).tolist()
    test_y = np.load(data_dir / "test_y.npy").astype(np.int32).tolist()
    
    print(f"Loaded {len(train_x)} training samples, {len(test_x)} test samples")
    
    # Create output directory
    output_dir = script_dir / "modal_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Determine run name and output filename
    if weight_decay > 0:
        run_name = f"Weight Decay={weight_decay}"
    else:
        run_name = "No Weight Decay"
    
    output_filename = f"reference_training_loss{output_suffix}.csv"
    
    print("\n" + "="*70)
    print(f"Training: {run_name}")
    print("="*70)
    
    result = train_model.remote(
        train_x_list=train_x,
        train_y_list=train_y,
        test_x_list=test_x,
        test_y_list=test_y,
        hidden_size=hidden_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        log_interval_epochs=log_interval_epochs,
        weight_decay=weight_decay,
        run_name=run_name,
    )
    
    # Save result to CSV
    csv_path = output_dir / output_filename
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy', 'weight_norm', 'time_seconds'])
        for i in range(len(result['epochs'])):
            writer.writerow([
                result['epochs'][i],
                result['train_losses'][i],
                result['test_losses'][i],
                result['train_accuracies'][i],
                result['test_accuracies'][i],
                result['weight_norms'][i],
                result['times'][i],
            ])
    print(f"Saved to {csv_path}")
    
    # Copy to data directory for frontend
    import shutil
    final_path = data_dir / output_filename
    shutil.copy(csv_path, final_path)
    print(f"Copied to {final_path}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Training time: {result['total_time']:.2f}s ({result['total_time']/60:.2f} min)")
    print(f"Final train accuracy: {result['train_accuracies'][-1]:.4f}")
    print(f"Final test accuracy: {result['test_accuracies'][-1]:.4f}")
