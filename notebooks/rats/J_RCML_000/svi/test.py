from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import Predictive, SVI, Trace_ELBO

def model(data):
    f = numpyro.sample("latent_fairness", dist.Beta(10, 10))
    with numpyro.plate("N", data.shape[0] if data is not None else 10):
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

def guide(data):
    alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)
    beta_q = numpyro.param("beta_q", lambda rng_key: random.exponential(rng_key),
                        constraint=constraints.positive)
    numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))


data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
optimizer = numpyro.optim.Adam(step_size=0.0005)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

svi_result = svi.run(random.PRNGKey(0), 2000, data)
params = svi_result.params
inferred_mean = params["alpha_q"] / (params["alpha_q"] + params["beta_q"])

# use guide to make predictive
predictive = Predictive(model, guide=guide, params=params, num_samples=1000)
samples = predictive(random.PRNGKey(1), data=None)

print(type(samples))
print(list(samples.keys()))
print(samples['obs'].shape)
print(samples['obs'].mean(axis=0))

# get posterior samples
predictive = Predictive(guide, params=params, num_samples=1000)
posterior_samples = predictive(random.PRNGKey(1), data=None)

print(type(posterior_samples))
print(list(posterior_samples.keys()))
print(posterior_samples['latent_fairness'].shape)
print(posterior_samples['latent_fairness'].mean())

# use posterior samples to make predictive
predictive = Predictive(model, posterior_samples, params=params, num_samples=1000)
samples = predictive(random.PRNGKey(1), data=None)
print(type(samples))
print(list(samples.keys()))
print(samples['obs'].shape)
print(samples['obs'].mean(axis=0))
