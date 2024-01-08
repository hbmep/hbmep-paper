import functools
import os
import pickle
import logging
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

import arviz as az
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site

from simulate_data import Simulator

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def main():
    toml_path = "/home/vishu/repos/hbmep-paper/configs/experiments/subjects.toml"

    """ Load simulated data """
    dir ="/home/vishu/repos/hbmep-paper/reports/experiments/subjects/simulate-data/a_random_mean_-2.5_a_random_scale_1.5/"
    src = os.path.join(dir, "simulation_ppd.pkl")
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    dest = os.path.join(simulator.build_dir, "plot-simulated-data-log.log")
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

    for k, v in simulation_ppd.items():
        logger.info(f"{k}: {v.shape}")

    src = os.path.join(dir, "simulation_df.csv")
    simulation_df = pd.read_csv(src)

    obs = simulation_ppd[site.obs]
    a = simulation_ppd[site.a]
    logger.info(f"obs: {obs.shape}")
    logger.info(f"a: {a.shape}")
    valid_draws = (a > 0).all(axis=(1, 2, 3))
    logger.info(f"valid_draws: {valid_draws.shape}")
    logger.info(f"valid_draws: {valid_draws.sum()}")

    # """ Plot """
    # N_DRAWS_TO_PLOT = 10
    # temp_ppd = {
    #     k: v[:N_DRAWS_TO_PLOT, ...] for k, v in simulation_ppd.items()
    # }
    # assert temp_ppd[site.obs].shape[0] == N_DRAWS_TO_PLOT

    # temp_df = simulation_df.copy()
    # temp_ppd = {
    #     k: v.swapaxes(0, -1) for k, v in temp_ppd.items()
    # }
    # obs = temp_ppd[site.obs]
    # logger.info(f"obs: {obs.shape}")
    # response = [simulator.response[0] + f"_{i}" for i in range(N_DRAWS_TO_PLOT)]
    # temp_df[response] = obs[0, ...]
    # simulator.render_recruitment_curves(
    #     df=temp_df,
    #     response=response,
    #     response_colors = plt.cm.rainbow(np.linspace(0, 1, N_DRAWS_TO_PLOT)),
    #     prediction_df=temp_df,
    #     posterior_predictive=temp_ppd,
    #     posterior_samples=temp_ppd
    # )


if __name__ == "__main__":
    main()
