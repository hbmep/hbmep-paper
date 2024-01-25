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
    # Priors for the model parameters
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfCauchy(scale=5))

    # Linear relationship
    mean = relu(x, alpha, beta, 0)

    # Likelihood
    with numpyro.plate('data', x.shape[0]):
        numpyro.sample('obs', dist.Normal(mean, sigma), obs=y)


def guide_manual(x, y=None):
    # Variational parameters for alpha
    alpha_loc = numpyro.param('alpha_loc', 0.0)
    alpha_scale = numpyro.param('alpha_scale', 1.0, constraint=dist.constraints.positive)
    numpyro.sample('alpha', dist.Normal(alpha_loc, alpha_scale))

    # Variational parameters for beta
    beta_loc = numpyro.param('beta_loc', 0.0)
    beta_scale = numpyro.param('beta_scale', 1.0, constraint=dist.constraints.positive)
    numpyro.sample('beta', dist.Normal(beta_loc, beta_scale))

    # Variational parameters for sigma
    sigma_loc = numpyro.param('sigma_loc', 1.0, constraint=dist.constraints.positive)
    numpyro.sample('sigma', dist.HalfCauchy(scale=sigma_loc))


# Generating synthetic data
def generate_data(x, alpha_real, beta_real):

    noise = np.random.normal(scale=1.0, size=len(x))  # noise
    # y = alpha_real + beta_real * x + noise  # linear relationship with noise
    y = relu(x, alpha_real, beta_real, 0) + noise
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
x, y = generate_data(np.linspace(0, 10, 200),2.0, 2.5)

rng_key = random.PRNGKey(0)

optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
guide = guide_manual
# guide = numpyro.infer.autoguide.AutoNormal(model)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
n_steps = 1000
in_loop = True
if in_loop:
    svi_state = svi.init(rng_key, x, y)
    svi_state, loss = svi_step(svi_state, x, y)  # single step for JIT
    start_time = time.time()
    for step in range(n_steps):
        svi_state, loss = svi_step(svi_state, x, y)
    end_time = time.time()
    samples = return_samples(model, guide, svi.get_params(svi_state), x)

else:
    start_time = time.time()
    svi_result = svi.run(random.PRNGKey(0), n_steps, x, y, progress_bar=False)
    end_time = time.time()
    samples = return_samples(model, guide, svi_result.params, x)
    svi_state = svi_result.state

print(f'SVI:{end_time - start_time}')

nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=1000, num_chains=1, progress_bar=False)
# Run the MCMC
start_time = time.time()
mcmc.run(rng_key, x, y)
end_time = time.time()
print(f'MCMC:{end_time - start_time}')

n = 10
for i in range(n):
    plt.plot(x, samples['obs'][i, :], 'ro')

plt.plot(x, y, 'k')
plt.show()
#
# Now give it a change in the relation and see how it updates
x_, y_ = generate_data(np.linspace(4, 5, 3),2.5, 4.0)
# svi_state = svi.init(rng_key, x, y)
new_optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
svi.optim = new_optimizer  # I AM GUESSING HERE NOT SURE IF THIS DOES IT...

x_eval = np.linspace(0, 10, 25)
for step in range(400):
    svi_state, loss = svi_step(svi_state, x_, y_)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss}")
        samples_updated = return_samples(model, guide, svi.get_params(svi_state), x_eval)
        plt.plot(x_eval, samples_updated['obs'][0, :], 'o')
samples_updated = return_samples(model, guide, svi.get_params(svi_state), x_eval)

plt.plot(x, y, 'k')
plt.plot(x_, y_, 'k', markersize=8)
# plt.show()
