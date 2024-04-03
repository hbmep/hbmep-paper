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
from jax.scipy.signal import convolve
from numpyro.infer import init_to_feasible

def rate(mu, c_1, c_2):
    return (
            c_1 + jnp.true_divide(c_2, mu)
    )


def concentration(mu, beta):
    return jnp.multiply(mu, beta)


@jit
def svi_step(svi_state, x, y, t):
    svi_state, loss = svi.stable_update(svi_state, x, y, t)
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

    a_bio1, b_bio1 = 11, 2.50
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
    mu_bio1 = np.array(mu_bio1)
    sigma_bio1 = 0.5
    Y_rc1 = mu_bio1 + mu_bio1 * np.random.randn(*x.shape) * sigma_bio1
    Y_bio1 = signal_bio1 * Y_rc1

    for ix in range(input_size):
        d_max = 8
        sat = 80.
        if x[ix][0] > sat:
            d = int(d_max)
        else:
            d = int(np.round((x[ix][0]/sat) * d_max))
        Y_rolled_row = np.roll(Y_bio1[ix, :], -d, axis=0)
        Y_bio1[ix, :] = Y_rolled_row

    Y_noiseless = Y_bio1
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
    # noise_bio1 = numpyro.sample("noise_bio1", dist.LogNormal(0.0, 50.0))
    # length_bio1 = numpyro.sample("length_bio1", dist.LogNormal(0.0, 50.0))
    # kernel_bio1 = kernel(t, t, 1.0, length_bio1, noise_bio1)
    # gp_bio1 = numpyro.sample("gp_bio1", dist.MultivariateNormal(loc=jnp.zeros(t.shape[0]),
    #                                                             covariance_matrix=kernel_bio1))

    d = len(t)
    theta = numpyro.sample("theta", dist.HalfCauchy(jnp.ones(d)))
    con_chol = jnp.ones(1) * 0.5

    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(d, con_chol))
    sigma = jnp.sqrt(theta)
    L_Omega = jnp.matmul(jnp.diag(sigma), L_omega)
    mu = jnp.zeros(d)
    gp_bio1 = numpyro.sample("gp_bio1", dist.MultivariateNormal(mu, scale_tril=L_Omega))

    # corr_mat = numpyro.sample("corr_mat", dist.LKJ(d, con_chol))
    # sigma = jnp.sqrt(theta)
    # cov_mat = jnp.matmul(jnp.matmul(jnp.diag(sigma), corr_mat), jnp.diag(sigma))
    # mu = jnp.zeros(d)
    # gp_bio1 = numpyro.sample("gp_bio1", dist.MultivariateNormal(mu, covariance_matrix=cov_mat))

    b_bio1 = numpyro.sample("b_bio1", dist.HalfNormal(10))
    a_bio1 = numpyro.sample("a_bio1", dist.Normal(50, 100))

    L = numpyro.sample("L", dist.HalfNormal(1))

    mu_bio1 = F.relu(X.flatten()[:, None], a_bio1, b_bio1, L)   # you need +ve L or the obs model goes to nan I think

    c_1 = numpyro.sample('c_1', dist.HalfNormal(2.))
    c_2 = numpyro.sample('c_2', dist.HalfNormal(2.))
    beta = numpyro.deterministic('beta', rate(mu_bio1, c_1, c_2))
    alpha = numpyro.deterministic('alpha', concentration(mu_bio1, beta))
    draws_bio1 = numpyro.sample('draws_bio1', dist.Gamma(concentration=alpha, rate=beta))

    scaled_bio1 = draws_bio1 * gp_bio1

    scaled_response = scaled_bio1

    obs_noise = numpyro.sample("obs_noise", dist.HalfNormal(scale=1))

    numpyro.sample("Y", dist.Normal(scaled_response, obs_noise), obs=Y)

rng_key = random.PRNGKey(0)
N = 64  # Number of stimulation trials
T = 100  # Number of time points in the MEP time series
np.random.seed(0)
Y, X, t, Y_noiseless = generate_synthetic_data(T, N, noise_level=3.0)

framework = "SVI"
if framework == "MCMC":
    nuts_kernel = NUTS(model, init_strategy=init_to_feasible)
    mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=1000)
    mcmc.run(jax.random.PRNGKey(0), X, t, Y)
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
plt.plot(t, (k * X + Y).transpose(), 'r')
for ix_X in range(0, len(X), 4):
    x = X[ix_X]
    offset = x * k
    y_bio1 = offset + F.relu(x, ps['a_bio1'], ps['b_bio1'], ps['L']).reshape(-1, 1) * ps['gp_bio1']
    y_bio1 = y_bio1.transpose()

    for ix in range(10):
        plt.plot(t, y_bio1[:, ix], 'k')

plt.show()

plt.figure()
y_bio1 = ps['gp_bio1']
y_bio1 = y_bio1.transpose()
plt.plot(t, y_bio1, 'k')

plt.show()
