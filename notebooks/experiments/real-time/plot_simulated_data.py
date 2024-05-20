import os
import pickle
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from hbmep.model.utils import Site as site

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

SIMULATION_DIR = "/home/mcintosh/Local/temp/test_hbmep/hbmep_sim/build/simulate_data"
# SIM_TYPE = "existing_participants"
# SIM_TYPE = "existing_participants_equi_spaced_pulses"
SIM_TYPE = "new_participants"

SIMULATION_DF_PATH = os.path.join(SIMULATION_DIR, f"simulation_df_{SIM_TYPE}.csv")
SIMULATION_PPD_PATH = os.path.join(SIMULATION_DIR, f"simulation_ppd_{SIM_TYPE}.pkl")


def main():
    """ Load simulated data / ppd """
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    """ Set up logging in build directory """
    dest = os.path.join(simulator.build_dir, "plot_simulated_data.log")
    logging.basicConfig(
        format=FORMAT,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {dest}")

    ppd_obs = simulation_ppd[site.obs]
    ppd_a = simulation_ppd[site.a]
    logger.info(f"ppd_obs: {ppd_obs.shape}")
    logger.info(f"ppd_a: {ppd_a.shape}")

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

    dest = os.path.join(simulator.build_dir, f"{SIM_TYPE}.pdf")
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


if __name__ == "__main__":
    main()
