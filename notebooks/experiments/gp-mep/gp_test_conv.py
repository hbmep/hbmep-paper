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
from jax import vmap
jax.config.update("jax_enable_x64", True)


def rate(mu, c_1, c_2):
    return (
            c_1 + jnp.true_divide(c_2, mu)
    )

def convolve_wrapper(gp_bio1_core, f):
    return jnp.convolve(gp_bio1_core, f, mode='same')


convolve_vectorized = vmap(convolve_wrapper, in_axes=(None, 0))

def concentration(mu, beta):
    return jnp.multiply(mu, beta)


@jit
def svi_step(svi_state, x, y, t):
    svi_state, loss = svi.stable_update(svi_state, x, y, t)
    return svi_state, loss


def generate_filter_stack(time_range, shift, variance):
    filter_stack = jnp.exp(-0.5 * (time_range - shift) ** 2 / variance).T
    filter_stack = filter_stack - jnp.mean(filter_stack, axis=1, keepdims=True)  # Subtract the mean from each 'f'
    # k = 2  # Example shape parameter; adjust based on desired peakiness
    # sigma = jnp.sqrt(variance)  # Width parameter derived from variance
    # filter_stack = jnp.exp(-((time_range - shift) / sigma) ** (2 * k)).T
    return filter_stack

def generate_synthetic_data(seq_length, input_size, noise_level=0.25):
    x = np.linspace(0, 100, input_size).reshape(-1, 1)  # stim intensities
    t = np.linspace(0, 10, seq_length)  # Time points of the MEP response

    a_bio1, b_bio1 = 15, 0.150
    v_bio1, ell_bio1, H_bio1 = 1, 1, 50
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
    mu_bio1 = F.rectified_logistic(x, a_bio1, b_bio1, v_bio1, 0, ell_bio1, H_bio1)
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
        # if np.random.rand() > 0.5:
        #     Y_rolled_row = np.roll(Y_rolled_row, -np.random.randint(10), axis=0)
        # if np.random.rand() > 0.5:
        #     Y_rolled_row = - Y_rolled_row

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
    scaled_bios = jnp.zeros_like(t.shape[0])
    b_bio_parent = numpyro.sample(f"b_bio_parent", dist.Gamma(2, 1))
    H_bio_parent = numpyro.sample(f"H_bio_parent", dist.Gamma(2, 0.25))
    for i in range(0, n_bio):
        # Sample core GP for each bio component
        # need to think about why you need the noise in these GPs..
        noise_bio = numpyro.sample(f"noise_bio{i}", dist.LogNormal(0.0, 5.0))
        length_bio = numpyro.sample(f"length_bio{i}", dist.LogNormal(0.0, 5.0))
        kernel_bio = kernel(t, t, 1.0, length_bio, noise_bio)
        gp_bio_core = numpyro.sample(f"gp_bio_core{i}",
                                     dist.MultivariateNormal(loc=jnp.zeros(t.shape[0]), covariance_matrix=kernel_bio))
        gp_norm = numpyro.deterministic(f"gp_norm{i}", jnp.sum((gp_bio_core[1:] + gp_bio_core[:-1]) / 2))
        # gp_bio_norm = numpyro.deterministic(f"gp_bio_norm{i}", gp_bio_core / gp_norm)  # works but... much worse

        noise_shift = numpyro.sample(f"noise_shift{i}", dist.LogNormal(0.0, 25.0))
        length_shift = numpyro.sample(f"length_shift{i}", dist.LogNormal(0.0, 25.0))
        variance_shift = numpyro.sample(f"variance_shift{i}", dist.LogNormal(0.0, 5.0))
        kernel_shift = kernel(X[:, 0], X[:, 0], variance_shift, length_shift, noise_shift)
        shift = numpyro.sample(f"shift{i}", dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]),
                                                                covariance_matrix=kernel_shift))
        variance = numpyro.deterministic(f'variance{i}', v_global)
        filter_stack = generate_filter_stack(time_range[:, None], shift, variance)
        gp_bio = convolve_vectorized(gp_bio_core, filter_stack)

        # Scale each gp_bio component
        b_bio = numpyro.sample(f"b_bio{i}", dist.Gamma(0.25, b_bio_parent))
        # truncated_laplace = dist.LeftTruncatedDistribution(dist.Laplace(loc=0.0, scale=5.0), low=0.0)
        # b_bio = numpyro.sample(f"b_bio{i}", truncated_laplace, sample_shape=(X.shape[1],))

        a_bio = numpyro.sample(f"a_bio{i}", dist.Normal(50, 100))
        # L = numpyro.sample(f"L{i}", dist.HalfNormal(1))
        # mu_bio = F.relu(X.flatten()[:, None], a_bio, b_bio, 1e-6)
        v_bio = numpyro.sample(f"v_bio{i}", dist.HalfNormal(10))
        ell_bio = numpyro.sample(f"ell_bio{i}", dist.HalfNormal(10))
        H_bio = numpyro.sample(f"H_bio{i}", dist.Gamma(0.25, H_bio_parent))
        mu_bio = F.rectified_logistic(X.flatten()[:, None], a_bio, b_bio, v_bio, 1e-6, ell_bio, H_bio)

        c_1 = numpyro.sample(f'c_1_{i}', dist.HalfNormal(2.))
        c_2 = numpyro.sample(f'c_2_{i}', dist.HalfNormal(2.))
        beta = numpyro.deterministic(f'beta_{i}', rate(mu_bio, c_1, c_2))
        alpha = numpyro.deterministic(f'alpha_{i}', concentration(mu_bio, beta))
        draws_bio = numpyro.sample(f'draws_bio{i}', dist.Gamma(concentration=alpha, rate=beta))

        scaled_bio = draws_bio * gp_bio

        # Accumulate scaled_bio components
        scaled_bios += scaled_bio

    scaled_response = scaled_bios

    obs_noise = numpyro.sample("obs_noise", dist.HalfNormal(scale=1))
    numpyro.sample("Y", dist.Normal(scaled_response, obs_noise), obs=Y)

rng_key = random.PRNGKey(0)
N = 32  # Number of stimulation trials
T = 50  # Number of time points in the MEP time series

np.random.seed(0)
Y, X, t, Y_noiseless = generate_synthetic_data(T, N, noise_level=3.0)
# variance = 0.5  # n.b. this is a global
# means = np.linspace(t[0], t[-1], int(np.round((t[-1] - t[0]) / np.sqrt(variance))))  # n.b. this is a global
dt = np.median(np.diff(t))
n_bio = 2
time_range = jnp.array(np.arange(-dt * 20, dt * 20 + dt, dt))
v_global = np.square(dt * 1.5)

cmap = plt.cm.get_cmap('viridis', n_bio)
colors = [cmap(i) for i in range(n_bio)]

framework = "SVI"
num_samples = 200
if framework == "MCMC":
    nuts_kernel = NUTS(model, init_strategy=init_to_feasible)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=1000)
    mcmc.run(jax.random.PRNGKey(0), X, t, Y)
    ps = mcmc.get_samples()

elif framework == "SVI":
    optimizer = numpyro.optim.ClippedAdam(step_size=0.01)
    # guide = numpyro.infer.autoguide.AutoMultivariateNormal(model)
    # guide = numpyro.infer.autoguide.AutoLowRankMultivariateNormal(model)
    guide = numpyro.infer.autoguide.AutoNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=12))
    n_steps = int(5e4)
    svi_state = svi.init(rng_key, X, t, Y)
    print('SVI starting.')
    svi_state, loss = svi_step(svi_state, X, t, Y)  # single step for JIT
    print('JIT compile done.')
    for step in range(n_steps):
        svi_state, loss = svi_step(svi_state, X, t, Y)
        if step % 5000 == 0:
            predictive = Predictive(guide, params=svi.get_params(svi_state), num_samples=num_samples)
            ps = predictive(rng_key, X, t)
            # zero_row = jnp.zeros((ps['shift_core'].shape[0], 1))
            # ps['shift'] = jnp.concatenate([zero_row, ps['shift_core']], axis=1)
            predictive_obs = Predictive(model, ps, params=svi.get_params(svi_state), num_samples=num_samples)
            ps_obs = predictive_obs(rng_key, X, t)
            print(step)
            for ix_bio in range(0, n_bio):
                print(f"b{ix_bio}:{np.mean(ps[f'b_bio{ix_bio}'])}")
                print(f"H{ix_bio}:{np.mean(ps[f'H_bio{ix_bio}'])}")
            k = 10.0
            plt.figure()
            plt.plot(t, (k * X + Y).transpose(), 'r')
            for ix_X in range(0, len(X), 3):
                x = X[ix_X]
                offset = x * k
                variance_local = v_global
                # variance_local = ps['variance'][:]

                # if np.array(jnp.any(jnp.isinf(f))):
                #     continue
                for ix_bio in range(0, n_bio):
                    for ix_draw in range(0, ps[f"shift{ix_bio}"].shape[0], 5):
                        plt.plot(t, offset + ps_obs['Y'][ix_draw, ix_X, :], color='green')
                    plt.plot(ps[f"shift{ix_bio}"].transpose() + 2, X * k, color=colors[ix_bio])
            plt.show()

            plt.figure()
            for ix_bio in range(0, n_bio):
                plt.plot(X, ps[f"draws_bio{ix_bio}"].squeeze().transpose(), 'o', color=colors[ix_bio])
            plt.show()

            plt.figure()
            for ix_X in range(0, len(X), 3):
                x = X[ix_X]
                offset = x * 1 * 0.08
                variance_local = v_global
                # variance_local = ps['variance'][:]
                for ix_bio in range(0, n_bio):
                    filter_stack = generate_filter_stack(time_range[:, None], ps[f"shift{ix_bio}"][:, ix_X], variance_local)
                    for ix_draw in range(0, ps[f"shift{ix_bio}"].shape[0], 5):
                        plt.plot(time_range, offset + filter_stack[ix_draw, :], color=colors[ix_bio])
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
