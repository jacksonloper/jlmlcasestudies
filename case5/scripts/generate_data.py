"""
Generate data for Case Study 5: Likelihood Estimation

Generates:
- X = (X1, X2) in R^2 with X1, X2 ~ N(0, 1)
- Y | X=(x1,x2) is an even mixture of N(x1, 1) and N(x2, 1)

The true log likelihood of Y | X is:
  log p(y | x1, x2) = log(0.5 * N(y; x1, 1) + 0.5 * N(y; x2, 1))

Users must estimate this log likelihood for each test point.
Evaluation: MSE of log likelihoods.

Outputs:
- train_x.npy: 5000x2 matrix (X1, X2 pairs) as float32
- train_y.npy: 5000 vector (Y values) as float32
- test_x.npy: 500x2 matrix (X1, X2 pairs) as float32
- test_y.npy: 500 vector (Y values) as float32
- test_true_loglik.npy: 500 vector (true log likelihoods) as float32
"""

import numpy as np
from scipy.stats import norm
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_train = 5000
n_test = 500
n_total = n_train + n_test


def sample_data(n):
    """Sample n data points from the generative model."""
    # X = (X1, X2) ~ N(0, 1) each
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    # Y | X is mixture: 0.5 * N(x1, 1) + 0.5 * N(x2, 1)
    y = np.zeros(n)
    for i in range(n):
        if np.random.rand() < 0.5:
            y[i] = np.random.normal(x1[i], 1)
        else:
            y[i] = np.random.normal(x2[i], 1)

    x = np.column_stack([x1, x2])
    return x, y


def true_log_likelihood(x, y):
    """
    Compute true log p(y | x1, x2) for each data point.

    p(y | x1, x2) = 0.5 * N(y; x1, 1) + 0.5 * N(y; x2, 1)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]

    # Use logsumexp for numerical stability
    log_p1 = norm.logpdf(y, loc=x1, scale=1.0)
    log_p2 = norm.logpdf(y, loc=x2, scale=1.0)

    # log(0.5 * exp(log_p1) + 0.5 * exp(log_p2))
    # = log(0.5) + logsumexp(log_p1, log_p2)
    log_half = np.log(0.5)
    max_log = np.maximum(log_p1, log_p2)
    log_lik = log_half + max_log + np.log(np.exp(log_p1 - max_log) + np.exp(log_p2 - max_log))

    return log_lik


# Generate data
print("Generating training data...")
train_x, train_y = sample_data(n_train)

print("Generating test data...")
test_x, test_y = sample_data(n_test)

# Compute true log likelihoods for test set
print("Computing true log likelihoods for test set...")
test_true_loglik = true_log_likelihood(test_x, test_y)

# Create output directory
output_dir = Path(__file__).parent.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

# Save as float32
np.save(output_dir / "train_x.npy", train_x.astype(np.float32))
np.save(output_dir / "train_y.npy", train_y.astype(np.float32))
np.save(output_dir / "test_x.npy", test_x.astype(np.float32))
np.save(output_dir / "test_y.npy", test_y.astype(np.float32))
np.save(output_dir / "test_true_loglik.npy", test_true_loglik.astype(np.float32))

print(f"\nGenerated data saved to {output_dir}")
print(f"Train X shape: {train_x.shape}")
print(f"Train Y shape: {train_y.shape}")
print(f"Test X shape: {test_x.shape}")
print(f"Test Y shape: {test_y.shape}")
print(f"Test true log-likelihood shape: {test_true_loglik.shape}")
print(f"\nData statistics:")
print(f"  Train X1: mean={train_x[:, 0].mean():.2f}, std={train_x[:, 0].std():.2f}")
print(f"  Train X2: mean={train_x[:, 1].mean():.2f}, std={train_x[:, 1].std():.2f}")
print(f"  Train Y:  mean={train_y.mean():.2f}, std={train_y.std():.2f}")
print(f"  Test true log-lik: mean={test_true_loglik.mean():.2f}, std={test_true_loglik.std():.2f}")
print(f"  Test true log-lik: min={test_true_loglik.min():.2f}, max={test_true_loglik.max():.2f}")
