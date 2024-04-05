import numpy as np
from numpyro.infer import MCMC, NUTS
from hbmep.model import functional as F
from numpyro.infer import Predictive, SVI, Trace_ELBO
import jax
from jax import jit
from matplotlib import pyplot as plt
from numpyro.infer import init_to_feasible
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
jax.config.update("jax_enable_x64", True)


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
    sigma_bio1 = 0.1
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


def gaussian_basis_vectorized(time_range, means, variance):
    # Expand dimensions for broadcasting
    # time_range: [T] -> [T, 1]
    # means: [M] -> [1, M]
    # Output shape: [T, M]
    return jnp.exp(-0.5 * (time_range[:, None] - means[None, :]) ** 2 / variance)


def model(X, t, Y=None):
    noise_bio1 = numpyro.sample("noise_bio1", dist.LogNormal(0.0, 50.0))
    length_bio1 = numpyro.sample("length_bio1", dist.LogNormal(0.0, 50.0))
    kernel_bio1 = kernel(t, t, 1.0, length_bio1, noise_bio1)
    gp_bio1_core = numpyro.sample("gp_bio1_core", dist.MultivariateNormal(loc=jnp.zeros(t.shape[0]), covariance_matrix=kernel_bio1))
    gp_bio1 = jnp.zeros((N, len(t)))  # Placeholder for the samples

    noise_shift = numpyro.sample("noise_shift", dist.LogNormal(0.0, 10.0))
    length_shift = numpyro.sample("length_shift", dist.LogNormal(0.0, 10.0))
    variance_shift = numpyro.sample("variance_shift", dist.LogNormal(0.0, 10.0))
    kernel_shift = kernel(X[:, 0], X[:, 0], variance_shift, length_shift, noise_shift)
    shift = numpyro.sample("shift",
                                  dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=kernel_shift))
    # shift = numpyro.sample("shift", dist.Normal(jnp.zeros(N), 1 * jnp.ones(N)))
    # variance = numpyro.sample("variance", dist.Laplace(0.0, 100))
    # variance = numpyro.sample("variance", dist.LogNormal(0.0, 0.2))
    variance = numpyro.deterministic('variance', v_global)

    for i in range(N):
        f = jnp.exp(-0.5 * (time_range[:] - shift[i]) ** 2 / variance)
        f = f - jnp.mean(f)
        gp_bio1 = gp_bio1.at[i].set(jnp.convolve(gp_bio1_core, f, mode='same'))
    b_bio1 = numpyro.sample("b_bio1", dist.HalfNormal(10))
    a_bio1 = numpyro.sample("a_bio1", dist.Normal(50, 100))

    L = numpyro.sample("L", dist.HalfNormal(1))

    mu_bio1 = F.relu(X.flatten()[:, None], a_bio1, b_bio1, L)  # you need +ve L or the obs model goes to nan I think

    c_1 = numpyro.sample('c_1', dist.HalfNormal(2.))
    c_2 = numpyro.sample('c_2', dist.HalfNormal(2.))
    beta = numpyro.deterministic('beta', rate(mu_bio1, c_1, c_2))
    alpha = numpyro.deterministic('alpha', concentration(mu_bio1, beta))
    draws_bio1 = numpyro.sample('draws_bio1', dist.Gamma(concentration=alpha, rate=beta))

    scaled_bio1 = draws_bio1 * gp_bio1

    scaled_response = scaled_bio1

    obs_noise = numpyro.sample("obs_noise", dist.HalfNormal(scale=1))
    Y = numpyro.sample("Y", dist.Normal(scaled_response, obs_noise), obs=Y)

rng_key = random.PRNGKey(0)
N = 32  # Number of stimulation trials
T = 50  # Number of time points in the MEP time series

np.random.seed(0)
Y, X, t, Y_noiseless = generate_synthetic_data(T, N, noise_level=3.0)
# variance = 0.5  # n.b. this is a global
# means = np.linspace(t[0], t[-1], int(np.round((t[-1] - t[0]) / np.sqrt(variance))))  # n.b. this is a global
time_range = jnp.array(np.arange(-10, 10 + 1, 1))
dt = np.median(np.diff(t))
time_range = jnp.array(np.arange(-dt * 15, dt * 15 + dt, dt))
v_global = 0.1

framework = "SVI"
num_samples = 200
if framework == "MCMC":
    nuts_kernel = NUTS(model, init_strategy=init_to_feasible)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=1000)
    mcmc.run(jax.random.PRNGKey(0), X, t, Y)
    ps = mcmc.get_samples()

elif framework == "SVI":
    optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
    guide = numpyro.infer.autoguide.AutoNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    n_steps = int(1e5)
    svi_state = svi.init(rng_key, X, t, Y)
    print('SVI starting.')
    svi_state, loss = svi_step(svi_state, X, t, Y)  # single step for JIT
    print('JIT compile done.')
    for step in range(n_steps):
        svi_state, loss = svi_step(svi_state, X, t, Y)
        if step % 2000 == 0:
            predictive = Predictive(guide, params=svi.get_params(svi_state), num_samples=num_samples)
            ps = predictive(rng_key, X, t)
            predictive_obs = Predictive(model, ps, params=svi.get_params(svi_state), num_samples=num_samples)
            ps_obs = predictive_obs(rng_key, X, t)
            print(step)
            k = 10.0
            plt.figure()
            plt.plot(t, (k * X + Y).transpose(), 'r')
            for ix_X in range(0, len(X), 3):
                x = X[ix_X]
                offset = x * k
                variance_local = v_global
                # variance_local = ps['variance'][:]
                f = jnp.exp(-0.5 * ((time_range[:, None] - ps['shift'][:, ix_X]) ** 2) / variance_local)
                f = f - jnp.mean(f)
                if np.array(jnp.any(jnp.isinf(f))):
                    continue
                for ix_draw in range(0, ps['shift'].shape[0], 5):
                    # gp_bio1 = jnp.convolve(ps['gp_bio1_core'][ix_draw, :], f[:, ix_draw], mode='same')
                    # y_bio1 = offset + F.relu(x, ps['a_bio1'][ix_draw], ps['gp_bio1_core'][ix_draw], ps['L'][ix_draw]) * gp_bio1
                    # plt.plot(t, y_bio1, 'k')

                    plt.plot(t, offset + ps_obs['Y'][ix_draw, ix_X, :], color='green')
            plt.plot(ps['shift'].transpose() + 2, X * k, color='blue')
            plt.show()

            plt.figure()
            for ix_X in range(0, len(X), 3):
                x = X[ix_X]
                offset = x * 1 * 0.08
                variance_local = v_global
                # variance_local = ps['variance'][:]
                f = jnp.exp(-0.5 * ((time_range[:, None] - ps['shift'][:, ix_X]) ** 2) / variance_local)
                f = f - jnp.mean(f)
                for ix_draw in range(0, ps['shift'].shape[0], 5):
                    plt.plot(time_range, offset + f[:, ix_draw], color='blue')
            plt.show()
            print(1)
    print('SVI done.')

else:
    raise Exception("?")



# plt.figure()
# C_matrices = []
#
# for ix in range(ps['L_omega'].shape[0]):
#     # Compute L_Omega_scaled as before
#     L_Omega_scaled = jnp.matmul(jnp.diag(jnp.sqrt(ps['theta'][ix, :])), ps['L_omega'][ix, 0, :, :])
#     # Compute C as before
#     C = jnp.matmul(L_Omega_scaled, L_Omega_scaled.T)
#
#     # Zero out the diagonal
#     diagonal_mask = jnp.eye(C.shape[0], dtype=bool)
#     C = C.at[diagonal_mask].set(0)
#
#     # Append the computed C to the list
#     C_matrices.append(C)
#
# # Stack the collected C matrices along a new third dimension
# C_stacked = jnp.stack(C_matrices, axis=-1)
#
# C = jnp.mean(C_stacked, 2)
# plt.imshow(C)
# plt.show()
