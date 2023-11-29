import os
import pickle
import logging
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, median_absolute_error

import jax
import jax.numpy as jnp

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.experiments.tms.sparse_subjects_power import HierarchicalBayesianSimulator

logger = logging.getLogger(__name__)


def _process(arr):
    me = arr.mean(axis=-1)
    sem = stats.sem(arr, axis=-1)
    std = arr.std(axis=-1)
    return me, sem, std


def _process_bp(arr):
    arr = ((arr < 0).mean(axis=(-1, -2, -3)) > .95).mean(axis=-1)
    return _process(arr)


def _process_fp(arr):
    arr = (arr < .05).mean(axis=-1)
    return _process(arr)


if __name__=="__main__":
    toml_path = "/home/vishu/repos/hbmep-paper/configs/paper/tms/mixed-effects/simulator/hierarchical_bayesian_simulator.toml"
    mu_a_delta, sigma_a_delta = -1.5, 1
    simulation_prefix = f"mu_a_delta_{mu_a_delta}__sigma_a_delta_{sigma_a_delta}"

    CONFIG = Config(toml_path=toml_path)
    CONFIG.BUILD_DIR = os.path.join(CONFIG.BUILD_DIR, simulation_prefix)

    SIMULATOR = HierarchicalBayesianSimulator(config=CONFIG, mu_a_delta=mu_a_delta, sigma_a_delta=sigma_a_delta)

    N_space = [1, 2, 4, 6, 8, 12, 16, 20]
    n_draws = 50
    n_repeats = 50

    # N_LIM, DRAW_LIM, SEED_LIM = len(N_space), 14, n_repeats
    # fname = f"N_LIM_{N_LIM}__DRAW_LIM_{DRAW_LIM}__SEED_LIM_{SEED_LIM}"
    # extension = ".pkl"

    src = "/home/vishu/cpu12/N_LIM_8__DRAW_LIM_START_15_END_29__SEED_LIM_50.pkl"
    with open(src, "rb") as f:
        hbm_mu_delta, hbm_error, hbm_p_value, nhbm_error, nhbm_p_value, = pickle.load(f)

    """ Plot """
    lw = 1.4
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False, constrained_layout=True)

    logger.info(hbm_error.shape)

    ax = axes[0, 0]
    me, sem, std = _process(hbm_error[..., 0].reshape(hbm_error.shape[0], -1))
    ax.errorbar(x=N_space, y=me, yerr=sem, marker="o", label="Hieararchical Bayesian", linestyle="--", linewidth=lw, ms=4)

    me, sem, std = _process(nhbm_error[..., 0].reshape(hbm_error.shape[0], -1))
    ax.errorbar(x=np.array(N_space) + .1, y=me, yerr=sem, marker="o", label="Non-Hieararchical Bayesian", linestyle="--", linewidth=lw, ms=4)

    ax.set_xticks(N_space)
    ax.grid()
    ax.legend()

    ax = axes[0, 1]
    me, sem, std = _process_bp(hbm_mu_delta)
    ax.errorbar(x=N_space, y=me, yerr=sem, marker="o", label="Hiearchical Bayesian", linestyle="--", linewidth=lw, ms=4)

    me, sem, std = _process_fp(nhbm_p_value[..., 1])
    ax.errorbar(x=np.array(N_space), y=me, yerr=sem, marker="o", label="Non-Hieararchical Bayesian", linestyle="--", linewidth=lw, ms=4)

    ax.set_xticks(N_space)
    ax.set_yticks(np.arange(0, 1.1, .1))
    ax.grid()
    ax.legend()

    dest = "/home/vishu/cpu12/cpu12.png"
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")
