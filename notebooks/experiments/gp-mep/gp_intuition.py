import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from jax import random
from jax.numpy import linspace
from jax.scipy.linalg import cho_solve, cho_factor
from numpy.random import multivariate_normal
import jax
from jax import numpy as jnp
# from jax.scipy.linalg import cholesky
from jax.scipy.linalg import cholesky

from jax.scipy.stats import multivariate_normal as jax_mvn
import jax.config

# Set JAX to use 64-bit floating point numbers by default
jax.config.update("jax_enable_x64", True)

def rbf_kernel(x1, x2, variance=1.0, length_scale=1.0):
    """Compute the RBF kernel between two sets of vectors."""
    sqdist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
    return variance * np.exp(-0.5 / length_scale ** 2 * sqdist)


def draw_samples_from_gp_jax(X, y, Xnew, kernel_function, num_samples=10, noise=0.25):
    """Draw samples from a GP posterior for new input points Xnew using JAX for improved performance."""
    pred_mean, pred_cov = gaussian_process(X, y, Xnew, kernel_function, noise=noise)

    # Make sure the covariance matrix is symmetric positive-semidefinite
    pred_cov = (pred_cov + pred_cov.T) / 2
    pred_cov += 1e-6 * jnp.eye(len(Xnew))  # Add a small value to the diagonal for numerical stability

    # Cholesky decomposition for sampling
    L = cholesky(pred_cov, lower=True)

    # Generate samples from a standard normal distribution
    z = jax.random.normal(jax.random.PRNGKey(0), (num_samples, len(Xnew)))

    # Transform standard normal samples using the Cholesky decomposition
    samples = pred_mean.T + jnp.dot(L, z.T).T

    return samples


def draw_samples_from_gp(X, y, Xnew, kernel_function, num_samples=10, noise=0.25):
    """Draw samples from a GP posterior for new input points Xnew."""
    pred_mean, pred_cov = gaussian_process(X, y, Xnew, kernel_function, noise=noise)

    # Ensure the covariance matrix is positive-semidefinite
    pred_cov = (pred_cov + pred_cov.T) / 2 + 1e-6 * np.eye(len(Xnew))

    # Draw samples from the multivariate normal distribution
    samples = multivariate_normal(pred_mean.flatten(), pred_cov, num_samples)

    return samples


def gaussian_process(X, y, Xnew, kernel_function, noise=0.25):
    """
    Manually implement Gaussian Process regression.

    Parameters:
    - X: np.array, shape (N, D)
        The input features for training data, where N is the number of samples
        and D is the number of features.
    - y: np.array, shape (N, 1)
        The target values for training data.
    - Xnew: np.array, shape (M, D)
        The input features for which predictions are to be made.
    - kernel_function: function
        The kernel function used for computing the covariance matrix. It should
        take two arrays as input and return a covariance matrix.
    - noise: float, default 0.25
        The noise level in the data. It is added to the diagonal of the kernel
        matrix to make it positive definite and to represent the variance of the
        observation noise.

    Returns:
    - pred_mean: np.array, shape (M, 1)
        The mean of the predictive distribution for each input in Xnew.
    - pred_var: np.array, shape (M,)
        The variance of the predictive distribution for each input in Xnew.
    """
    # Compute the kernel matrix K for the training data, and add noise variance
    K = kernel_function(X, X) + noise ** 2 * np.eye(len(X))

    # Compute the kernel matrix between training data and new data points
    K_s = kernel_function(X, Xnew)

    # Compute the kernel matrix for the new data points
    K_ss = kernel_function(Xnew, Xnew)

    # Factorize the kernel matrix K (Cholesky decomposition) for solving
    L = cho_factor(K)

    # Solve the linear system for the weights (alpha)
    alpha = cho_solve(L, y)

    # Compute the predictive mean by multiplying the transpose of K_s by alpha
    pred_mean = np.dot(K_s.T, alpha)

    # Compute the predictive variance
    v = cho_solve(L, K_s)  # Solve for covariance between training and new data
    pred_var = K_ss - np.dot(K_s.T, v)  # Variance is reduced by observed data

    return pred_mean, np.diag(pred_var)


# Set random seed for reproducibility
np.random.seed(0)

# Data preparation
X = np.array([-10, -8, -5, -1, 1, 3, 7, 10]).reshape(-1, 1)
n_extra = 20
X = np.vstack([X, np.random.rand(n_extra, 1) * 20 - 10])
X = np.sort(X, axis=0)
N = X.shape[0]
y_noiseless = 1 - X * 5e-2 + np.sin(X) / X
y = y_noiseless + 0.25 * np.random.randn(N, 1)
M = 500
Xnew = linspace(-10, 10, M).reshape(-1, 1)

# Compute predictions using the defined GPR
pred_mean, pred_var = gaussian_process(X, y, Xnew, rbf_kernel)

# Plot the predictions with confidence intervals
plt.figure(figsize=(10, 6))

# Plot mean prediction
plt.plot(Xnew.flatten(), pred_mean.flatten(), 'k', lw=2, zorder=9)
plt.fill_between(Xnew.flatten(),
                 pred_mean.flatten() - 1.96 * np.sqrt(pred_var),
                 pred_mean.flatten() + 1.96 * np.sqrt(pred_var),
                 alpha=0.2, color='gray')

plt.scatter(X, y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
plt.title("Gaussian Process Regression Prediction with Confidence Interval")
plt.xlabel("X")
plt.ylabel("Prediction")

plt.show()

# Draw samples using the JAX-based function
num_samples = 3
samples = draw_samples_from_gp(X, y, Xnew, rbf_kernel, num_samples=num_samples)


# Plot the samples along with the mean prediction and confidence interval
plt.figure(figsize=(10, 6))
for i in range(num_samples):
    plt.plot(Xnew.flatten(), samples[i, :], lw=1, ls='--', label=f'Sample {i + 1}')

# Plot mean prediction and confidence interval
plt.plot(Xnew.flatten(), pred_mean.flatten(), 'k', lw=2, zorder=9, label='Mean Prediction')
plt.fill_between(Xnew.flatten(),
                 pred_mean.flatten() - 1.96 * np.sqrt(pred_var),
                 pred_mean.flatten() + 1.96 * np.sqrt(pred_var),
                 alpha=0.2, color='gray', label='95% Confidence Interval')

plt.scatter(X, y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
plt.title("Gaussian Process Regression with Samples")
plt.xlabel("X")
plt.ylabel("Prediction")
plt.legend()
plt.show()

