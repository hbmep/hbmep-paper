import os
import functools
import pickle
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jax.numpy as jnp

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from constants import (
    TOTAL_SUBJECTS,
    TOTAL_PULSES,
    TOTAL_REPS,
    REP
)

logger = logging.getLogger(__name__)
SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-speed/simulate_data"
SIMULATION_DF_PATH = os.path.join(SIMULATION_DIR, "simulation_df.csv")
SIMULATION_PPD_PATH = os.path.join(SIMULATION_DIR, "simulation_ppd.pkl")
BUILD_DIR = SIMULATION_DIR


def plot(simulator, simulation_df, simulation_ppd, n_draws_to_plot, dest):
    """ Plot """
    temp_ppd = {
        k: v[:n_draws_to_plot, ...] for k, v in simulation_ppd.items()
    }
    assert temp_ppd[site.obs].shape[0] == n_draws_to_plot

    temp_df = simulation_df.copy()
    temp_ppd = {
        k: v.swapaxes(0, -1) for k, v in temp_ppd.items()
    }

    temp_ppd_obs = temp_ppd[site.obs]
    logger.info(f"temp_ppd_obs: {temp_ppd_obs.shape}")

    response = [simulator.response[0] + f"_{i}" for i in range(n_draws_to_plot)]
    temp_df[response] = temp_ppd_obs[0, ...]

    simulator.render_recruitment_curves(
        df=temp_df,
        response=response,
        response_colors = plt.cm.rainbow(np.linspace(0, 1, n_draws_to_plot)),
        prediction_df=temp_df,
        posterior_predictive=temp_ppd,
        posterior_samples=temp_ppd,
        destination_path=dest
    )
    logger.info(f"Saved to {dest}")
    return


def main():
    """ Load simulated data / ppd """
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    logger.info(f"simulation_df: {simulation_df.shape}")

    for r in [1, 2, 4, 8]:
        n_draws_to_plot = 5
        if r == 1: n_draws_to_plot = 50

        ind = simulation_df[REP] < r
        temp_simulation_df = simulation_df[ind].reset_index(drop=True).copy()
        temp_simulation_ppd = simulation_ppd.copy()
        for s in [site.mu, site.obs]:
            temp_simulation_ppd[s] = temp_simulation_ppd[s][:, ind, ...]

        dest = os.path.join(BUILD_DIR, f"reps_{r}.pdf")
        plot(simulator, temp_simulation_df, temp_simulation_ppd, n_draws_to_plot, dest)

    return


if __name__ == "__main__":
    main()