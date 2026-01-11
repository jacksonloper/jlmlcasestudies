"""
Train reference solution for Case Study 4: Modular Division using JAX on Modal.com with T4 GPU.

This script trains neural networks to learn modular division mod 97.
Architecture: 
- Input: 194 (one-hot encoding of a and b)
- MLP: 194 -> 256 (with ReLU)
- Multi-head attention layer (1 layer, treating 256 as sequence of tokens)
- Output: 256 -> 97 logits

Training: Cross-entropy loss with AdamW optimizer.

Outputs:
- reference_training_loss_wd.csv: Training history with weight decay
"""

import modal

# Create Modal app
app = modal.App("case4-reference-jax")

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
    timeout=90 * 60,  # 90 minute timeout
)
def train_model(train_x_list, train_y_list, test_x_list, test_y_list, 
                hidden_size=256, n_heads=4, n_epochs=50000, learning_rate=0.001, batch_size=512,
                log_interval_epochs=10, weight_decay=0.0, run_name="default"):
    """
    Train a neural network with MLP + attention for modular division using JAX.
    
    Architecture:
    - MLP: 194 -> 256 
    - Reshape to (num_tokens, head_dim) for attention
    - Multi-head attention (1 layer)
    - Pool and project to 97 logits
    
    Args:
        train_x_list: Training inputs as list of lists (one-hot encoded)
        train_y_list: Training labels as list
        test_x_list: Test inputs as list of lists (one-hot encoded)
        test_y_list: Test labels as list
        hidden_size: Size of hidden layer (default 256)
        n_heads: Number of attention heads (default 4)
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        batch_size: Minibatch size
        log_interval_epochs: Log metrics every N epochs (for CSV storage)
        weight_decay: Weight decay for AdamW
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
    
    # Attention configuration
    # We'll treat the hidden representation as a sequence
    # hidden_size should be divisible by n_heads
    head_dim = hidden_size // n_heads
    num_tokens = n_heads  # Reshape hidden as (n_heads, head_dim)
    
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
    print(f"Architecture: {input_dim} -> {hidden_size} (MLP) -> MultiHeadAttn({n_heads} heads, dim={head_dim}) -> {n_classes}")
    print(f"Logging every {log_interval_epochs} epochs")
    
    # Set random seed
    key = random.PRNGKey(42)
    
    def init_network_params(key):
        """Initialize MLP + attention network with Xavier initialization."""
        keys = random.split(key, 10)
        
        # MLP layer: input_dim -> hidden_size
        w_mlp = random.normal(keys[0], (input_dim, hidden_size)) * jnp.sqrt(2.0 / (input_dim + hidden_size))
        b_mlp = jnp.zeros(hidden_size)
        
        # Attention parameters
        # Query, Key, Value projections (applied to each token)
        # Each token has dimension head_dim, and we project to head_dim for Q, K, V
        w_q = random.normal(keys[1], (head_dim, head_dim)) * jnp.sqrt(2.0 / (2 * head_dim))
        w_k = random.normal(keys[2], (head_dim, head_dim)) * jnp.sqrt(2.0 / (2 * head_dim))
        w_v = random.normal(keys[3], (head_dim, head_dim)) * jnp.sqrt(2.0 / (2 * head_dim))
        
        # Output projection after attention (from hidden_size to hidden_size)
        w_o = random.normal(keys[4], (hidden_size, hidden_size)) * jnp.sqrt(2.0 / (2 * hidden_size))
        b_o = jnp.zeros(hidden_size)
        
        # Final output layer: hidden_size -> n_classes
        w_out = random.normal(keys[5], (hidden_size, n_classes)) * jnp.sqrt(2.0 / (hidden_size + n_classes))
        b_out = jnp.zeros(n_classes)
        
        return {
            'w_mlp': w_mlp, 'b_mlp': b_mlp,
            'w_q': w_q, 'w_k': w_k, 'w_v': w_v,
            'w_o': w_o, 'b_o': b_o,
            'w_out': w_out, 'b_out': b_out,
        }
    
    def multihead_attention(params, x):
        """
        Apply multi-head self-attention.
        
        x: (num_tokens, head_dim) - reshaped hidden representation
        Returns: (num_tokens, head_dim) - attended representation
        """
        # x shape: (num_tokens, head_dim)
        # Apply Q, K, V projections to each token
        Q = jnp.dot(x, params['w_q'])  # (num_tokens, head_dim)
        K = jnp.dot(x, params['w_k'])  # (num_tokens, head_dim)
        V = jnp.dot(x, params['w_v'])  # (num_tokens, head_dim)
        
        # Scaled dot-product attention
        # scores: (num_tokens, num_tokens)
        scale = jnp.sqrt(head_dim).astype(jnp.float32)
        scores = jnp.dot(Q, K.T) / scale
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Weighted sum of values
        # output: (num_tokens, head_dim)
        output = jnp.dot(attn_weights, V)
        
        return output
    
    def forward(params, x):
        """
        Forward pass: 
        1. MLP: input -> hidden
        2. Reshape to (num_tokens, head_dim)
        3. Multi-head attention
        4. Flatten and project to logits
        """
        # MLP layer
        h = jnp.dot(x, params['w_mlp']) + params['b_mlp']
        h = jax.nn.relu(h)  # (hidden_size,)
        
        # Reshape for attention: (hidden_size,) -> (num_tokens, head_dim)
        h_reshaped = h.reshape(num_tokens, head_dim)
        
        # Apply multi-head attention
        h_attn = multihead_attention(params, h_reshaped)  # (num_tokens, head_dim)
        
        # Flatten back: (num_tokens, head_dim) -> (hidden_size,)
        h_flat = h_attn.reshape(hidden_size)
        
        # Output projection with residual connection
        h_out = jnp.dot(h_flat, params['w_o']) + params['b_o']
        h_out = h_out + h  # Residual connection
        h_out = jax.nn.relu(h_out)
        
        # Final logits
        logits = jnp.dot(h_out, params['w_out']) + params['b_out']
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
        for key in ['w_mlp', 'w_q', 'w_k', 'w_v', 'w_o', 'w_out']:
            total_norm_sq += jnp.sum(params[key] ** 2)
        return jnp.sqrt(total_norm_sq)
    
    def compute_weight_norms_by_layer(params):
        """Compute L2 norms for each layer group separately.
        
        Returns:
            mlp1_norm: Norm of the input MLP layer (w_mlp)
            attn_norm: Norm of the attention weights (w_q, w_k, w_v)
            mlp2_norm: Norm of the output projection and final layer (w_o, w_out)
        """
        # MLP1: Input MLP layer
        mlp1_norm_sq = jnp.sum(params['w_mlp'] ** 2)
        mlp1_norm = jnp.sqrt(mlp1_norm_sq)
        
        # Attention: Q, K, V projections
        attn_norm_sq = (jnp.sum(params['w_q'] ** 2) + 
                        jnp.sum(params['w_k'] ** 2) + 
                        jnp.sum(params['w_v'] ** 2))
        attn_norm = jnp.sqrt(attn_norm_sq)
        
        # MLP2: Output projection and final layer
        mlp2_norm_sq = jnp.sum(params['w_o'] ** 2) + jnp.sum(params['w_out'] ** 2)
        mlp2_norm = jnp.sqrt(mlp2_norm_sq)
        
        return mlp1_norm, attn_norm, mlp2_norm
    
    # Training loop
    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"{'Epoch':>8} {'Train Loss':>12} {'Test Loss':>12} {'Train Acc':>10} {'Test Acc':>10} {'W Norm':>10} {'MLP1':>8} {'Attn':>8} {'MLP2':>8} {'Time':>8}")
    print("-" * 110)
    
    start_time = time.time()
    
    epochs_recorded = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    weight_norms = []
    weight_norms_mlp1 = []
    weight_norms_attn = []
    weight_norms_mlp2 = []
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
            mlp1_norm, attn_norm, mlp2_norm = compute_weight_norms_by_layer(params)
            
            train_loss = float(train_loss)
            test_loss = float(test_loss)
            train_acc = float(train_acc)
            test_acc = float(test_acc)
            weight_norm = float(weight_norm)
            mlp1_norm = float(mlp1_norm)
            attn_norm = float(attn_norm)
            mlp2_norm = float(mlp2_norm)
            
            epochs_recorded.append(epoch)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            weight_norms.append(weight_norm)
            weight_norms_mlp1.append(mlp1_norm)
            weight_norms_attn.append(attn_norm)
            weight_norms_mlp2.append(mlp2_norm)
            times_recorded.append(elapsed)
            
            print(f"{epoch:8d} {train_loss:12.4f} {test_loss:12.4f} {train_acc:10.4f} {test_acc:10.4f} {weight_norm:10.2f} {mlp1_norm:8.2f} {attn_norm:8.2f} {mlp2_norm:8.2f} {elapsed:8.1f}s")
            
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
        'weight_norms_mlp1': weight_norms_mlp1,
        'weight_norms_attn': weight_norms_attn,
        'weight_norms_mlp2': weight_norms_mlp2,
        'times': times_recorded,
        'total_time': total_time,
        'run_name': run_name,
    }


@app.local_entrypoint()
def main(hidden_size: int = 256, n_heads: int = 4, n_epochs: int = 50000, learning_rate: float = 0.001, 
         batch_size: int = 512, log_interval_epochs: int = 10, weight_decay: float = 0.0,
         output_suffix: str = ""):
    """
    Main entrypoint for running training on Modal.
    
    Args:
        hidden_size: Size of hidden layer (default 256)
        n_heads: Number of attention heads (default 4)
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
    
    print(f"Starting Case 4 (Modular Division) training on Modal with T4 GPU...")
    print(f"Architecture: 194 -> {hidden_size} (MLP) -> MultiHeadAttn({n_heads} heads) -> 97")
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
        n_heads=n_heads,
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
        writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy', 'weight_norm', 'weight_norm_mlp1', 'weight_norm_attn', 'weight_norm_mlp2', 'time_seconds'])
        for i in range(len(result['epochs'])):
            writer.writerow([
                result['epochs'][i],
                result['train_losses'][i],
                result['test_losses'][i],
                result['train_accuracies'][i],
                result['test_accuracies'][i],
                result['weight_norms'][i],
                result['weight_norms_mlp1'][i],
                result['weight_norms_attn'][i],
                result['weight_norms_mlp2'][i],
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
