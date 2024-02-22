import os
import pickle
import logging

import arviz as az
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

from jax import random
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi

from hbmep.config import Config
from hbmep.nn import functional as F

from hbmep_paper.utils import setup_logging
# from models_archived import Current


logger = logging.getLogger(__name__)

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/tms/config.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"

FEATURES = [["participant", "participant_condition"]]
RESPONSE = ['PKPK_ADM', 'PKPK_APB', 'PKPK_Biceps', 'PKPK_ECR', 'PKPK_FCR', 'PKPK_Triceps']

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/testing/"

import jax.numpy as jnp
import jax

def main():
    params = (4., 2., 1., .1, 1., 1)
    L = params[3]
    c1, c2 = .005, .05
    x = np.linspace(0, 10, 1000)
    mu = F.rectified_logistic(x, *params)
    beta = (1 / c1) + (1 / (c2 * mu))
    alpha = mu * beta
    # alpha = c1 * mu+ c2

    samples = dist.Gamma(concentration=alpha, rate=beta).sample(random.PRNGKey(0), (4000,))
    interval = hpdi(samples, prob=0.95)
    logger.info(samples.shape)

    nrows, ncols = 4, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 6), constrained_layout=True, squeeze=False)
    ax = axes[0, 0]
    sns.lineplot(x=x, y=mu, ax=ax)
    sns.lineplot(x=x, y=samples.mean(axis=0), ax=ax, color="red",alpha=.5)
    ax.fill_between(x, interval[0], interval[1], alpha=0.3)
    # sns.lineplot(x=x, y=samples.max(axis=0), color="g", ax=ax)
    ax = axes[1, 0]
    sns.lineplot(x=x, y=alpha, ax=ax)
    ax.set_title("Alpha")
    ax = axes[2, 0]
    scale = 1/ beta
    sns.lineplot(x=x, y=scale, ax=ax)
    ax.set_title("Scale")
    ax = axes[3, 0]
    sns.lineplot(x=x, y=beta, ax=ax)
    ax.set_title("Beta")

    dest = os.path.join(BUILD_DIR, "test.png")
    fig.savefig(dest)
    logger.info(dest)

    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname="test",
    )
    main()
