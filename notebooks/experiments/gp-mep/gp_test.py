import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from hbmep.model import functional as F
from numpyro.infer import Predictive, SVI, Trace_ELBO
import jax
from jax import random
from jax import jit
from matplotlib import pyplot as plt


def rate(mu, c_1, c_2):
    return (
            c_1 + jnp.true_divide(c_2, mu)
    )


def concentration(mu, beta):
    return jnp.multiply(mu, beta)


@jit
def svi_step(svi_state, x, y, t):
    svi_state, loss = svi.update(svi_state, x, y, t)
    return svi_state, loss


# def return_samples(model, guide, params, x, t):
#     predictive = Predictive(guide, params=params, num_samples=1000)
#     posterior_samples = predictive(random.PRNGKey(1), x, t)
#     predictive = Predictive(model, posterior_samples, params=params, num_samples=1000)
#     samples = predictive(random.PRNGKey(1), x, t)
#     return samples

def generate_synthetic_data(seq_length, input_size, noise_level=0.25):
    x = np.linspace(0, 100, input_size).reshape(-1, 1)  # stim intensities
    t = np.linspace(0, 10, seq_length)  # Time points of the MEP response

    a_bio1, b_bio1 = 22, 1.50
    a_bio2, b_bio2 = 66, 2.0

    b_art = 1.0
    # Generate a random Gaussian peak
    peak_position = int(0.25 * seq_length) + np.random.randint(seq_length - int(0.5 * seq_length))
    s1_time = (seq_length * (1 + np.random.rand()) / 40)
    s2_time = (seq_length * (1 + np.random.rand()) / 40)
    signal1 = + np.exp(-(np.arange(seq_length) - peak_position) ** 2 / (2 * s1_time ** 2))
    signal2 = - np.exp(-(np.arange(seq_length) - peak_position - s1_time) ** 2 / (2 * s2_time ** 2))
    s1 = (np.random.rand() - 0.5) * 15
    s2 = s1 + (np.random.rand() - 0.5) * 2
    signal_bio1 = signal1 * s1 + signal2 * s2
    signal_bio1 = signal_bio1 / np.abs(signal_bio1).max()  # just to help interpretation
    mu_bio1 = F.relu(x, a_bio1, b_bio1, 0)
    sigma_bio1 = 0.05
    Y_rc1 = mu_bio1 + mu_bio1 * np.random.randn(*x.shape) * sigma_bio1
    Y_bio1 = signal_bio1 * Y_rc1

    peak_position_pre = peak_position - (s1_time + s2_time) * 1.33
    s1_pre_time = s1_time / 4
    s2_pre_time = s2_time / 4
    signal1 = + np.exp(-(np.arange(seq_length) - peak_position_pre) ** 2 / (2 * s1_pre_time ** 2))
    signal2 = - np.exp(-(np.arange(seq_length) - peak_position_pre - s1_pre_time) ** 2 / (2 * s2_pre_time ** 2))
    s1_pre = (np.random.rand() - 0.5) * 15
    s2_pre = s1_pre + (np.random.rand() - 0.5) * 2
    signal_bio2 = signal1 * s1_pre + signal2 * s2_pre
    signal_bio2 = signal_bio2 / np.abs(signal_bio2).max()  # just to help interpretation
    Y_bio2 = signal_bio2 * F.relu(x, a_bio2, b_bio2, 0)

    signal_art = + np.exp(-(np.arange(seq_length) - 2) ** 2 / (2 * 1 ** 2))
    Y_art = signal_art * F.relu(x, 0, b_art, 0)

    Y_noiseless = Y_bio1 + Y_bio2 + Y_art
    Y_noise = np.random.normal(0, noise_level, Y_noiseless.shape)
    Y = Y_noiseless + Y_noise

    return Y, x, t, Y_noiseless


def kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


def model(X, t, Y=None):
    noise_bio1 = numpyro.sample("noise_bio1", dist.LogNormal(0.0, 10.0))
    length_bio1 = numpyro.sample("length_bio1", dist.LogNormal(0.0, 10.0))
    kernel_bio1 = kernel(t, t, 1.0, length_bio1, noise_bio1)
    gp_bio1 = numpyro.sample("gp_bio1", dist.MultivariateNormal(loc=jnp.zeros(t.shape[0]),
                                                                covariance_matrix=kernel_bio1))
    b_bio1 = numpyro.sample("b_bio1", dist.HalfNormal(10))
    a_bio1 = numpyro.sample("a_bio1", dist.Normal(50, 100))
    mu_bio1 = F.relu(X.flatten()[:, None], a_bio1, b_bio1, 0.1)   # you need +ve L or the obs model goes to nan I think

    c_1 = numpyro.sample('c_1', dist.HalfNormal(1.))
    c_2 = numpyro.sample('c_2', dist.HalfNormal(1.))
    beta = numpyro.deterministic('beta', rate(mu_bio1, c_1, c_2))
    alpha = numpyro.deterministic('alpha', concentration(mu_bio1, beta))
    draws_bio1 = numpyro.sample('draws_bio1', dist.Gamma(concentration=alpha, rate=beta))

    scaled_bio1 = draws_bio1 * gp_bio1

    noise_bio2 = numpyro.sample("noise_bio2", dist.LogNormal(0.0, 10.0))
    length_bio2 = numpyro.sample("length_bio2", dist.LogNormal(0.0, 10.0))
    kernel_bio2 = kernel(t, t, 1.0, length_bio2, noise_bio2)
    gp_bio2 = numpyro.sample("gp_bio2",
                             dist.MultivariateNormal(loc=jnp.zeros(t.shape[0]), covariance_matrix=kernel_bio2))
    b_bio2 = numpyro.sample("b_bio2", dist.HalfNormal(10))
    a_bio2 = numpyro.sample("a_bio2", dist.Normal(50, 100))
    scaled_bio2 = F.relu(X.flatten()[:, None], a_bio2, b_bio2, 0) * gp_bio2

    noise_art = numpyro.sample("noise_art", dist.LogNormal(0.0, 10.0))
    length_art = numpyro.sample("length_art", dist.LogNormal(0.0, 10.0))
    kernel_art = kernel(t, t, 1.0, length_art, noise_art)
    gp_art = numpyro.sample("gp_art",
                            dist.MultivariateNormal(loc=jnp.zeros(t.shape[0]), covariance_matrix=kernel_art))
    b_art = numpyro.sample("b_art", dist.HalfNormal(10))
    a_art = numpyro.sample("a_art", dist.Normal(50, 100))
    scaled_art = F.relu(X.flatten()[:, None], a_art, b_art, 0) * gp_art

    L = numpyro.sample("L", dist.HalfNormal(1))

    scaled_response = L + scaled_bio1 + scaled_bio2 + scaled_art

    obs_noise = numpyro.sample("obs_noise", dist.HalfNormal(scale=1))

    numpyro.sample("Y", dist.Normal(scaled_response, obs_noise), obs=Y)

rng_key = random.PRNGKey(0)
N = 64  # Number of stimulation trials
T = 100  # Number of time points in the MEP time series
np.random.seed(0)
Y, X, t, Y_noiseless = generate_synthetic_data(T, N, noise_level=2.0)

framework = "SVI"
if framework == "MCMC":
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=1000)
    mcmc.run(jax.random.PRNGKey(0), X, Y, t)
    ps = mcmc.get_samples()

elif framework == "SVI":
    optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
    guide = numpyro.infer.autoguide.AutoNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    n_steps = int(2e4)
    svi_state = svi.init(rng_key, X, t, Y)
    svi_state, loss = svi_step(svi_state, X, t, Y)  # single step for JIT
    for step in range(n_steps):
        svi_state, loss = svi_step(svi_state, X, t, Y)
    predictive = Predictive(guide, params=svi.get_params(svi_state), num_samples=1000)
    ps = predictive(random.PRNGKey(1), X, t)

else:
    raise Exception("?")

k = 10.0
plt.figure()
for ix_X in range(len(X)):
    x = X[ix_X]
    offset = x * k
    y_bio1 = offset + F.relu(x, ps['a_bio1'], ps['b_bio1'], ps['L']).reshape(-1, 1) * ps['gp_bio1']
    y_bio1 = y_bio1.transpose()
    y_bio2 = offset + F.relu(x, ps['a_bio2'], ps['b_bio2'], ps['L']).reshape(-1, 1) * ps['gp_bio2']
    y_bio2 = y_bio2.transpose()
    y_art = offset + F.relu(x, 0, ps['b_art'], 0).reshape(-1, 1) * ps['gp_art']
    y_art = y_art.transpose()
    for ix in range(10):
        plt.plot(t, y_bio1[:, ix], 'k')
        plt.plot(t, y_bio2[:, ix], 'b')
        plt.plot(t, y_art[:, ix], 'r')
plt.show()

plt.figure()
y_bio1 = ps['gp_bio1']
y_bio1 = y_bio1.transpose()
y_bio2 = ps['gp_bio2']
y_bio2 = y_bio2.transpose()
y_art = ps['gp_art']
y_art = y_art.transpose()
plt.plot(t, y_bio1, 'k')
plt.plot(t, y_bio2, 'b')
plt.plot(t, y_art, 'r')
plt.show()

plt.figure()
plt.plot((k * X + Y).transpose())
plt.show()
