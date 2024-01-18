import os
import pickle
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging

logger = logging.getLogger(__name__)

# SIMULATION_DF_PATH = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/a_random_mean_-2.5_a_random_scale_1.5/simulation_df.csv"
# SIMULATION_PPD_PATH = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/a_random_mean_-2.5_a_random_scale_1.5/simulation_ppd.pkl"
# FILTER_PATH = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/a_random_mean_-2.5_a_random_scale_1.5/filter.npy"
# BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/a_random_mean_-2.5_a_random_scale_1.5/"

# SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_-3.0_a_random_scale_1.5/old"
SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_-3.0_a_random_scale_1.5"

SIMULATION_DF_PATH = os.path.join(SIMULATION_DIR, "simulation_df.csv")
SIMULATION_PPD_PATH = os.path.join(SIMULATION_DIR, "simulation_ppd.pkl")

BUILD_DIR = SIMULATION_DIR


def plot(simulator, simulation_df, simulation_ppd, dest):
    """ Plot """
    N_DRAWS_TO_PLOT = 3
    temp_ppd = {
        k: v[:N_DRAWS_TO_PLOT, ...] for k, v in simulation_ppd.items()
    }
    assert temp_ppd[site.obs].shape[0] == N_DRAWS_TO_PLOT

    temp_df = simulation_df.copy()
    temp_ppd = {
        k: v.swapaxes(0, -1) for k, v in temp_ppd.items()
    }

    temp_ppd_obs = temp_ppd[site.obs]
    logger.info(f"temp_ppd_obs: {temp_ppd_obs.shape}")

    response = [simulator.response[0] + f"_{i}" for i in range(N_DRAWS_TO_PLOT)]
    temp_df[response] = temp_ppd_obs[0, ...]

    simulator.render_recruitment_curves(
        df=temp_df,
        response=response,
        response_colors = plt.cm.rainbow(np.linspace(0, 1, N_DRAWS_TO_PLOT)),
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

    for k, v in simulation_ppd.items():
        logger.info(f"{k}: {v.shape}")

    TOTAL_SUBJECTS = 10
    TOTAL_PULSES = 101

    for r in [1, 2, 4, 8]:
        last_index = r * 2 * TOTAL_SUBJECTS * TOTAL_PULSES
        temp_sim_df = simulation_df[:last_index].reset_index(drop=True).copy()
        temp_ppd = simulation_ppd.copy()
        for s in [site.mu, site.obs]:
            temp_ppd[s] = temp_ppd[s][:, :last_index, ...]

        dest = os.path.join(BUILD_DIR, f"r{r}.pdf")
        plot(simulator, temp_sim_df, temp_ppd, dest)

    return


if __name__ == "__main__":
    main()
