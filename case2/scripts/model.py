"""
Shared rectified flow matching model for Case Study 2.

This module provides a reusable implementation of rectified flow matching
that can be configured with different feature extractors and architectures.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp


class RectifiedFlowModel:
    """
    Rectified Flow Matching model for learning conditional distributions.
    
    Args:
        hidden_layers: Tuple of hidden layer sizes (e.g., (128, 128, 64))
        feature_extractor: Function that takes (x, t, zt) and returns feature vector
        learning_rate: Learning rate for Adam optimizer
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, hidden_layers=(128, 128, 64), feature_extractor=None, 
                 learning_rate=0.001, random_state=42):
        self.hidden_layers = hidden_layers
        self.feature_extractor = feature_extractor or self._default_feature_extractor
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Scalers for standardization
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # MLP model (initialized during training)
        self.model = None
        
    def _default_feature_extractor(self, x, t, zt):
        """Default feature extractor: just raw values"""
        x = float(np.atleast_1d(x).flatten()[0])
        t = float(np.atleast_1d(t).flatten()[0])
        zt = float(np.atleast_1d(zt).flatten()[0])
        return np.array([x, t, zt])
    
    def fit_scalers(self, train_x, train_y):
        """Fit the input scalers on training data"""
        self.x_scaler.fit(train_x.reshape(-1, 1))
        self.y_scaler.fit(train_y.reshape(-1, 1))
        
    def transform_x(self, x):
        """Transform x using fitted scaler"""
        return self.x_scaler.transform(x.reshape(-1, 1)).flatten()
    
    def transform_y(self, y):
        """Transform y using fitted scaler"""
        return self.y_scaler.transform(y.reshape(-1, 1)).flatten()
    
    def inverse_transform_y(self, y_scaled):
        """Inverse transform y back to original scale"""
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def generate_flow_batch(self, x_data, y_data, n_t_per_sample=3):
        """
        Generate a batch of flow training data.
        
        For each sample, generates n_t_per_sample different t values:
        - t=0 (beginning)
        - t=1 (ending)
        - t~random (middle, repeated n_t_per_sample-2 times)
        
        Args:
            x_data: Array of x values (already scaled)
            y_data: Array of y values (already scaled)
            n_t_per_sample: Number of t values per sample (default: 3)
            
        Returns:
            X_batch: Feature matrix
            y_batch: Target values (y - eps)
        """
        n_samples = len(x_data)
        n_total = n_samples * n_t_per_sample
        
        # Vectorized: replicate each sample n_t_per_sample times
        x_expanded = np.repeat(x_data, n_t_per_sample)
        y_expanded = np.repeat(y_data, n_t_per_sample)
        
        # Vectorized: generate t values for all samples at once
        # For each sample: [0.0, 1.0, random, random, ...]
        t_values = np.zeros(n_total)
        t_values[1::n_t_per_sample] = 1.0  # Every 2nd element in each group
        # Fill remaining positions with random values
        for i in range(2, n_t_per_sample):
            t_values[i::n_t_per_sample] = np.random.uniform(0.0, 1.0, n_samples)
        
        # Vectorized: generate all random noise at once
        eps_values = np.random.randn(n_total)
        
        # Vectorized: compute all z_t values
        zt_values = y_expanded * t_values + (1 - t_values) * eps_values
        
        # Vectorized: compute all targets
        y_batch = y_expanded - eps_values
        
        # Create features using the feature extractor
        X_batch = []
        for i in range(n_total):
            features = self.feature_extractor(x_expanded[i], t_values[i], zt_values[i])
            X_batch.append(features)
        
        return np.array(X_batch), y_batch
    
    def _velocity_field(self, t, z, x_val):
        """Velocity field for the ODE: dz/dt = v(x, t, z)"""
        z_scalar = float(np.atleast_1d(z).flatten()[0])
        features = self.feature_extractor(x_val, t, z_scalar)
        v = self.model.predict(features.reshape(1, -1))[0]
        return [v]
    
    def generate_sample(self, x_val):
        """
        Generate a sample by integrating from z_0 ~ N(0,1) to z_1 ~ y
        
        Args:
            x_val: Scaled x value
            
        Returns:
            Scaled y sample
        """
        # Initial condition: z_0 ~ N(0,1)
        z0 = [np.random.randn()]
        
        # Time span for integration (from 0 to 1)
        t_span = (0, 1)
        
        # Solve ODE using RK45 with tighter tolerances
        try:
            solution = solve_ivp(
                lambda t, z: self._velocity_field(t, z, x_val),
                t_span,
                z0,
                method='RK45',
                rtol=1e-5,
                atol=1e-7,
                dense_output=True
            )
            
            if solution.success:
                return solution.y[0, -1]
            else:
                return z0[0]
        except (ValueError, RuntimeError):
            return z0[0]
    
    def train(self, train_x, train_y, n_epochs=300, n_t_per_sample=3, 
              val_split=0.1, patience=30, verbose=True):
        """
        Train the rectified flow model using partial_fit.
        
        Args:
            train_x: Training x values (unscaled)
            train_y: Training y values (unscaled)
            n_epochs: Number of training epochs
            n_t_per_sample: Number of t values per sample
            val_split: Fraction of data to use for validation
            patience: Early stopping patience (in units of 10 epochs)
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Fit scalers and transform data
        self.fit_scalers(train_x, train_y)
        train_x_scaled = self.transform_x(train_x)
        train_y_scaled = self.transform_y(train_y)
        
        # Split into train and validation
        n_train = len(train_x_scaled)
        n_val = int(val_split * n_train)
        val_indices = np.random.choice(n_train, n_val, replace=False)
        train_indices = np.array([i for i in range(n_train) if i not in val_indices])
        
        val_x = train_x_scaled[val_indices]
        val_y = train_y_scaled[val_indices]
        train_x_sub = train_x_scaled[train_indices]
        train_y_sub = train_y_scaled[train_indices]
        
        if verbose:
            print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
        
        # Initialize model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            random_state=self.random_state,
            learning_rate_init=self.learning_rate,
            solver='adam',
            max_iter=1
        )
        
        # Generate initial batch to initialize model structure
        X_init, y_init = self.generate_flow_batch(train_x_sub[:100], train_y_sub[:100], n_t_per_sample)
        self.model.partial_fit(X_init, y_init)
        
        # Fixed validation batch for consistent scoring
        VALIDATION_SEED = 999
        np.random.seed(VALIDATION_SEED)
        X_val_fixed, y_val_fixed = self.generate_flow_batch(val_x, val_y, n_t_per_sample)
        np.random.seed(self.random_state)
        
        best_val_loss = np.inf
        no_improve_count = 0
        
        # Track errors for history
        train_mse = []
        val_mse = []
        val_energy_scores = []
        epochs_recorded = []
        
        if verbose:
            print(f"Epoch    Train MSE    Val MSE    Val Energy Score")
            print("-" * 60)
        
        for epoch in range(n_epochs):
            # Generate fresh batch with new random t, eps for each training sample
            X_train, y_train = self.generate_flow_batch(train_x_sub, train_y_sub, n_t_per_sample)
            
            # Shuffle and train
            shuffle_idx = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_idx]
            y_train = y_train[shuffle_idx]
            
            # Update model with this epoch's batch
            self.model.partial_fit(X_train, y_train)
            
            # Evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                # Calculate MSE loss
                train_pred = self.model.predict(X_train)
                val_pred = self.model.predict(X_val_fixed)
                
                train_loss = np.mean((train_pred - y_train) ** 2)
                val_loss = np.mean((val_pred - y_val_fixed) ** 2)
                
                # Calculate energy score on validation set
                val_samples_scaled = np.zeros((len(val_x), 2), dtype=np.float32)
                for i in range(len(val_x)):
                    val_samples_scaled[i, 0] = self.generate_sample(val_x[i])
                    val_samples_scaled[i, 1] = self.generate_sample(val_x[i])
                
                # Transform back to original scale
                val_samples = self.inverse_transform_y(val_samples_scaled.flatten()).reshape(-1, 2)
                val_y_orig = self.inverse_transform_y(val_y)
                
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
                
                if verbose:
                    print(f"{epoch+1:5d}    {train_loss:10.4f}    {val_loss:8.4f}    {val_energy:14.4f}")
                
                # Early stopping based on validation MSE
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                if no_improve_count >= patience // 10:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'epochs': epochs_recorded,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'val_energy_scores': val_energy_scores
        }
    
    def predict_samples(self, test_x, n_samples=2, verbose=True):
        """
        Generate samples for test points.
        
        Args:
            test_x: Test x values (unscaled)
            n_samples: Number of samples to generate per test point
            verbose: Whether to print progress
            
        Returns:
            Array of shape (len(test_x), n_samples) with unscaled samples
        """
        test_x_scaled = self.transform_x(test_x)
        samples_scaled = np.zeros((len(test_x_scaled), n_samples), dtype=np.float32)
        
        for i in range(len(test_x_scaled)):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Generating samples {i+1}/{len(test_x_scaled)}...")
            
            x_val = test_x_scaled[i]
            for j in range(n_samples):
                samples_scaled[i, j] = self.generate_sample(x_val)
        
        # Transform back to original scale
        samples = self.inverse_transform_y(samples_scaled.flatten()).reshape(-1, n_samples)
        return samples.astype(np.float32)


def fourier_embedding(x, n_freqs=5):
    """Create Fourier features for input x"""
    x = np.atleast_1d(x).flatten()
    freqs = 2 ** np.arange(n_freqs)
    features = []
    for freq in freqs:
        features.append(np.sin(2 * np.pi * freq * x))
        features.append(np.cos(2 * np.pi * freq * x))
    return np.concatenate([f.flatten() for f in features])


def create_fourier_feature_extractor(n_freqs=5):
    """
    Create a feature extractor that uses Fourier embeddings.
    
    Args:
        n_freqs: Number of frequencies for Fourier embedding
        
    Returns:
        Feature extractor function
    """
    def feature_extractor(x, t, zt):
        """Create Fourier embeddings + raw features of (x, t, zt)"""
        x = float(np.atleast_1d(x).flatten()[0])
        t = float(np.atleast_1d(t).flatten()[0])
        zt = float(np.atleast_1d(zt).flatten()[0])
        
        # Create Fourier embeddings for each component
        x_emb = fourier_embedding(x, n_freqs=n_freqs)
        t_emb = fourier_embedding(t, n_freqs=n_freqs)
        zt_emb = fourier_embedding(zt, n_freqs=n_freqs)
        
        # Concatenate raw values + Fourier features
        return np.concatenate([[x, t, zt], x_emb, t_emb, zt_emb])
    
    return feature_extractor


def create_raw_feature_extractor():
    """
    Create a feature extractor that uses only raw features (no Fourier).
    
    Returns:
        Feature extractor function
    """
    def feature_extractor(x, t, zt):
        """Extract raw features only"""
        x = float(np.atleast_1d(x).flatten()[0])
        t = float(np.atleast_1d(t).flatten()[0])
        zt = float(np.atleast_1d(zt).flatten()[0])
        return np.array([x, t, zt])
    
    return feature_extractor


def calculate_energy_score(test_y, samples):
    """
    Calculate energy score for predictions.
    
    Args:
        test_y: True y values
        samples: Predicted samples (shape: (n_test, n_samples))
        
    Returns:
        Energy score
    """
    sum_dist1 = np.sum(np.abs(test_y - samples[:, 0]))
    sum_dist2 = np.sum(np.abs(test_y - samples[:, 1]))
    sum_dist_samples = np.sum(np.abs(samples[:, 0] - samples[:, 1]))
    
    energy_score = (sum_dist1 + sum_dist2) / (2 * len(test_y)) - 0.5 * sum_dist_samples / len(test_y)
    return energy_score
