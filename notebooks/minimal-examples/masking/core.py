from jax import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.enable_x64()
numpyro.set_host_device_count(24)

TOTAL_SUBJECTS = int(1e3)
MEASUREMENTS_PER_SUBJECT = 10


def simulate_data():
    # Fix rng seed
    rng_key = random.PRNGKey(1)

    # True parameters
    mu = 170
    sigma = 10

    # Generate true heights for each subject
    heights_true = dist.Normal(mu, sigma).sample(rng_key, (TOTAL_SUBJECTS,))
    heights_true = np.array(heights_true)

    # Add noise to true heights
    noise_sigma = 2
    heights_obs = dist.Normal(heights_true, noise_sigma).sample(rng_key, (MEASUREMENTS_PER_SUBJECT,))
    heights_obs = np.array(heights_obs)

    # Generate subject indices that match height_obs
    subject_idx = []
    for _ in range(MEASUREMENTS_PER_SUBJECT):
        for i in range(TOTAL_SUBJECTS):
            subject_idx.append(i)

    subject_idx = np.array(subject_idx)
    subject_idx = subject_idx.reshape(MEASUREMENTS_PER_SUBJECT, TOTAL_SUBJECTS)
    return heights_obs, heights_true, subject_idx


def _model(subject_idx, heights_mask, heights_obs=None):
    n_subjects = heights_mask.shape[0]
    n_obs = heights_obs.shape[0] if heights_obs is not None else -1

    mu = numpyro.sample('mu', dist.Normal(170, 10))
    sigma = numpyro.sample('sigma', dist.HalfCauchy(5))

    noise_sigma = numpyro.sample('noise_sigma', dist.HalfCauchy(5))

    with numpyro.plate('n_subjects', n_subjects):
        with numpyro.handlers.mask(mask=heights_mask):
            heights = numpyro.sample('heights', dist.Normal(mu, sigma))

    with numpyro.plate('n_obs', n_obs):
        numpyro.sample('obs', dist.Normal(heights[subject_idx], noise_sigma), obs=heights_obs)


def main(num_subjects_obs=1, num_subjects=10):
    # Simulate data
    heights_obs, heights_true, subject_idx = simulate_data()
    print(f"Shapes after simulating data:\nheights_obs: {heights_obs.shape}, subject_idx: {subject_idx.shape}")

    # We will give only the first subjects, but we tell the model that there are 10 subjects in totals
    # We will mask the parameters corresponding to the other 9 subjects

    # We will only give the first "num_subjects_obs" subjects
    heights_obs = heights_obs[..., :num_subjects_obs]
    subject_idx = subject_idx[..., :num_subjects_obs]
    print(f"Shapes after truncating data:\nheights_obs: {heights_obs.shape}, subject_idx: {subject_idx.shape}")

    # We will also flatten these arrays
    heights_obs = heights_obs.reshape(-1)
    subject_idx = subject_idx.reshape(-1)

    # Mask the heights of missing subjects
    heights_mask = [True for _ in range(num_subjects_obs)] + [False for _ in range(num_subjects_obs, num_subjects)]
    heights_mask = np.array(heights_mask)

    # Run inference
    sampler = NUTS(_model)
    mcmc_params = {
        'num_warmup': 1000000,
        'num_samples': 1000,
        'num_chains': 4,
    }
    rng_key = random.PRNGKey(0)

    mcmc = MCMC(sampler, **mcmc_params)
    mcmc.run(rng_key, subject_idx, heights_mask, heights_obs)

    posterior_samples = mcmc.get_samples()
    posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
    return mcmc, posterior_samples


if __name__ == "__main__":
    heights_obs, heights_true, subject_idx = simulate_data()

    num_subjects_obs = 1
    num_subjects = 10

    mcmc, posterior_samples = main(num_subjects_obs=num_subjects_obs, num_subjects=num_subjects)
    mcmc.print_summary()

    nrows, ncols = 1, 1
    fig, axes = plt.subp