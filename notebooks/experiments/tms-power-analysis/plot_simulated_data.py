import os
import pickle
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from constants import (
    REP,
    INFERENCE_FILE,
    SIMULATION_DF,
    SIMULATE_DATA_DIR,
    SIMULATE_DATA_NO_EFFECT_DIR
)

logger = logging.getLogger(__name__)


def plot(simulator, simulation_df, simulation_ppd, n_draws_to_plot, dest):
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


def main(build_dir):
    simulation_df_path = os.path.join(build_dir, SIMULATION_DF)
    simulation_ppd_path = os.path.join(build_dir, INFERENCE_FILE)

    # Load simulated data / ppd
    simulation_df = pd.read_csv(simulation_df_path)
    with open(simulation_ppd_path, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    # Set up logging
    setup_logging(
        dir=build_dir,
        fname=os.path.basename(__file__)
    )
    logger.info(f"simulation_df: {simulation_df.shape}")

    # Plot
    r = 1
    n_draws_to_plot = 50
    ind = simulation_df[REP] < r
    temp_simulation_df = simulation_df[ind].reset_index(drop=True).copy()
    temp_simulation_ppd = simulation_ppd.copy()
    for s in [site.mu, site.obs]:
        temp_simulation_ppd[s] = temp_simulation_ppd[s][:, ind, ...]

    dest = os.path.join(build_dir, f"plot_reps_{r}.pdf")
    plot(simulator, temp_simulation_df, temp_simulation_ppd, n_draws_to_plot, dest)

    return


if __name__ == "__main__":
    # Plot simulated data with effect
    main(SIMULATE_DATA_DIR)

    # Plot simulated data without effect
    main(SIMULATE_DATA_NO_EFFECT_DIR)

