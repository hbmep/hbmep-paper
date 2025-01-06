import os

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
    mu = np.array([170, 150])
    sigma = 10

    # Generate true heights for each subject
    heights_true = dist.Normal(mu, sigma).sample(rng_key, (TOTAL_SUBJECTS,))
    heights_true = np.array(heights_true)

    # Add noise to true heights
    noise_sigma = 2
    heights_obs = dist.Normal(heights_true, noise_sigma).sample(rng_key, (MEASUREMENTS_PER_SUBJECT,))
    heights_obs = np.array(heights_obs)

    return heights_obs, heights_true


def _model(subject_ind, group_ind, y_obs=None):
    n_subjects = np.max(subject_ind) + 1
    n_groups = np.max(group_ind) + 1
    n_data = subject_ind.shape[0]

    group_loc_loc = numpyro.sample('group_loc_loc', dist.Normal(165, 50))
    group_loc_scale = numpyro.sample('group_loc_scale', dist.HalfNormal(50))

    heights_scale = numpyro.sample('heights_scale', dist.HalfNormal(50))
    scale = numpyro.sample('scale', dist.HalfNormal(50))

    with numpyro.plate("n_groups", n_groups):
        group_loc = numpyro.sample('group_loc', dist.Normal(group_loc_loc, group_loc_scale))

        with numpyro.plate("n_subjects", n_subjects):
            heights = numpyro.sample('heights', dist.Normal(group_loc, heights_scale))

    with numpyro.plate('n_data', n_data):
        numpyro.sample('obs', dist.Normal(heights[subject_ind, group_ind], scale), obs=y_obs)


def _model_masked(subject_ind, group_ind, y_obs=None):
    n_subjects = np.max(subject_ind) + 1
    n_groups = np.max(group_ind) + 1
    n_data = subject_ind.shape[0]

    mask = np.full((n_subjects, n_groups), False)
    mask[subject_ind, group_ind] = True

    group_loc_loc = numpyro.sample('group_loc_loc', dist.Normal(165, 50))
    group_loc_scale = numpyro.sample('group_loc_scale', dist.HalfNormal(50))

    heights_scale = numpyro.sample('heights_scale', dist.HalfNormal(50))
    scale = numpyro.sample('scale', dist.HalfNormal(50))

    with numpyro.plate("n_groups", n_groups):
        group_loc = numpyro.sample('group_loc', dist.Normal(group_loc_loc, group_loc_scale))

        with numpyro.plate("n_subjects", n_subjects):
            with numpyro.handlers.mask(mask=mask):
                heights = numpyro.sample('heights', dist.Normal(group_loc, heights_scale))

    with numpyro.plate('n_data', n_data):
        numpyro.sample('obs', dist.Normal(heights[subject_ind, group_ind], scale), obs=y_obs)


def main(num_subjects=[10, 10], model=None):
    # Simulate data
    heights_obs, heights_true = simulate_data()
    assert len(num_subjects) == heights_obs.shape[-1]
    print(f"Shapes after simulating data:\nheights_obs: {heights_obs.shape}")

    # We will give only the first subjects, but we tell the model that there are 10 subjects in totals
    # We will mask the parameters corresponding to the other 9 subjects

    # We will only give the first "num_subjects_obs" subjects

    y_obs = []
    subject_ind = []
    group_ind = []

    for g in range(heights_obs.shape[-1]):
        num_participants_of_group = num_subjects[g]

        curr_y_obs = heights_obs[..., g]
        curr_y_obs = curr_y_obs[..., :num_participants_of_group]

        if not g: curr_subject_ind = np.arange(num_participants_of_group)
        else: curr_subject_ind = np.arange(num_participants_of_group) + np.sum(num_subjects[:g])

        curr_subject_ind = np.tile(curr_subject_ind, reps=MEASUREMENTS_PER_SUBJECT)
        curr_subject_ind = curr_subject_ind.reshape(MEASUREMENTS_PER_SUBJECT, -1)
        assert curr_subject_ind.shape == curr_y_obs.shape

        curr_y_obs = curr_y_obs.reshape(-1,)
        curr_subject_ind = curr_subject_ind.reshape(-1,)

        y_obs += curr_y_obs.tolist()
        subject_ind += curr_subject_ind.tolist()

        curr_group_ind = np.ones_like(curr_subject_ind) * g
        group_ind += curr_group_ind.tolist()

    y_obs = np.array(y_obs)
    subject_ind = np.array(subject_ind)
    group_ind = np.array(group_ind)

    # Run inference
    sampler = NUTS(model)
    mcmc_params = {
        'num_warmup': 4000,
        'num_samples': 1000,
        'num_chains': 4,
    }
    rng_key = random.PRNGKey(0)

    mcmc = MCMC(sampler, **mcmc_params)
    mcmc.run(rng_key, subject_ind, group_ind, y_obs)

    posterior_samples = mcmc.get_samples()
    posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
    return mcmc, posterior_samples


def plot():
    heights_estimated = posterior_samples['heights']
    group_loc = posterior_samples['group_loc']
    heights_scale = posterior_samples['heights_scale']
    group_loc_scale = posterior_samples['group_loc_scale']

    title = "without mask" if not ind else "masked"

    for g in range(len(num_subjects)):
        ax = axes[ind, g]
        x = np.arange(sum(num_subjects))
        y = heights_estimated[..., g]
        yme = y.mean(axis=0)
        yerr = y.std(axis=0)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=yerr,
            fmt='o',
            color="g",
            ms=8
        )

        x = np.arange(sum(num_subjects[:g]), sum(num_subjects[:g + 1]))
        yme = heights_true[..., g]
        yme = yme[:num_subjects[g]]
        ax.errorbar(
            x=x,
            y=yme,
            fmt='o',
            color="r"
        )
        ax.set_title(f"{title}: group {g}, num_subjects={num_subjects[g]}")

        ax = axes[-2, g]
        sns.kdeplot(group_loc[..., g], ax=ax, label=f"{title}")
        ax.set_title(f"group {g}, num_subjects={num_subjects[g]}")

    ax = axes[-1, 0]
    sns.kdeplot(heights_scale, ax=ax, label=f"{title}")
    ax.set_title("heights_scale")

    ax = axes[-1, 1]
    sns.kdeplot(group_loc_scale, ax=ax, label=f"{title}")
    ax.set_title("group_loc_scale")
    return axes


if __name__ == "__main__":
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(
        nrows, ncols, constrained_layout=True, squeeze=False,
        figsize=(ncols * 5, nrows * 2)
    )
    dest = os.path.join(os.getcwd(), 'group.png')

    heights_obs, heights_true = simulate_data()
    num_subjects = [10, 100]

    mcmc, posterior_samples = main(num_subjects=num_subjects, model=_model)
    mcmc.print_summary()
    ind = 0
    axes = plot()

    mcmc, posterior_samples = main(num_subjects=num_subjects, model=_model_masked)
    mcmc.print_summary()
    ind = 1
    axes = plot()

    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            ax.sharex(axes[0, 0])
            ax.sharey(axes[0, 0])

    ax = axes[0, 0]
    ax.set_xlim(-2, sum(num_subjects) + 2)
    ax.set_ylim(120, 200)

    axes[-2, 0].legend()

    fig.savefig(dest)
    print(f"Saved to {dest}")
