from jax import random
from jax import jit
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import Predictive, SVI, Trace_ELBO
import numpy as np
from matplotlib import pyplot as plt
import time
from numpyro.infer import MCMC, NUTS
# from hbmep.config import Config
# numpyro.set_host_device_count(14)

def relu(x, a, b, L):
    return (
        L
        + jnp.where(
            x <= a,
            0.,
            jnp.multiply(b, (x - a))
        )
    )


def model(x, y=None):
    n, num_regressions = x.shape  # Assuming x is a 2D array with shape (data_points, num_regressions)

    with numpyro.plate('regressions', num_regressions):
        alpha = numpyro.sample('alpha', dist.Normal(0, 10))
        beta = numpyro.sample('beta', dist.Normal(0, 10))
        sigma = numpyro.sample('sigma', dist.HalfCauchy(scale=5))

        mean = relu(x, alpha, beta, 0)

        with numpyro.plate('data', n):
            numpyro.sample('obs', dist.Normal(mean, sigma), obs=y)


# Generating synthetic data
def generate_data(num_points, num_regressions, alpha_real, beta_real):
    x_base = np.linspace(0, 10, num_points)
    x = np.tile(x_base[:, None], (1, num_regressions))  # Tile x for each regression

    y = np.zeros((num_points, num_regressions))

    for i in range(num_regressions):
        noise = np.random.normal(scale=1.0, size=num_points)
        alpha_real_ = alpha_real
        beta_real_ = beta_real * (1 + i/num_regressions)
        y[:, i] = relu(x[:, i], alpha_real_, beta_real_, 0) + noise

    return x, y


def return_samples(model, guide, params, x):
    predictive = Predictive(guide, params=params, num_samples=1000)
    posterior_samples = predictive(random.PRNGKey(1), x)
    predictive = Predictive(model, posterior_samples, params=params, num_samples=1000)
    samples = predictive(random.PRNGKey(1), x)
    return samples


@jit
def svi_step(svi_state, x, y):
    svi_state, loss = svi.update(svi_state, x, y)
    return svi_state, loss


np.random.seed(0)
num_models = 2500
x, y = generate_data(50, num_models, 2.0, 2.5)  # 200 data points, 100 independent regressions

rng_key = random.PRNGKey(0)

optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
# guide = guide_manual
guide = numpyro.infer.autoguide.AutoNormal(model)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
n_steps = 1000
svi_state = svi.init(rng_key, x, y)
svi_state, loss = svi_step(svi_state, x, y)  # single step for JIT
start_time = time.time()
for step in range(n_steps):
    svi_state, loss = svi_step(svi_state, x, y)
end_time = time.time()
samples = return_samples(model, guide, svi.get_params(svi_state), x)


print(f'SVI:{end_time - start_time}')
ix_model = int(num_models/2)
# ix_model = num_models - 1
n = 3
for i in range(n):
    plt.plot(x, samples['obs'][i, :, ix_model], 'ro')

plt.plot(x[:, ix_model], y[:, ix_model], 'k')
plt.show()