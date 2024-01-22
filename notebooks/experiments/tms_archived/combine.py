import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np
import jax

from hbmep.config import Config
from hbmep.model.utils import Site as site

from models import LearnPosterior, Simulator
from learn_posterior import TOML_PATH
from simulate_data import TOTAL_REPS, A_RANDOM_MEAN, A_RANDOM_SCALE
from hbmep_paper.utils import setup_logging

logger = logging.getLogger(__name__)


POSTERIOR_PATH = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/learn_posterior/inference.pkl"
BUILD_DIR = f"/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_{A_RANDOM_MEAN}_a_random_scale_{A_RANDOM_SCALE}/"

SIMULATION_DIR = f"/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_{A_RANDOM_MEAN}_a_random_scale_{A_RANDOM_SCALE}"
SIMULATION_DF = os.path.join(SIMULATION_DIR, "simulation_df.csv")
SIMULATION_PARAMS = os.path.join(SIMULATION_DIR, "simulation_params.pkl")


def combine(simulator, simulation_df, simulation_params):
    mu, beta, alpha, obs = [None] * 4
    for i in range(TOTAL_REPS):
        src = os.path.join(simulator.build_dir, f"simulation_ppd_{i}.pkl")
        with open(src, "rb") as f:
            simulation_ppd, = pickle.load(f)
        curr_mu, curr_beta, curr_alpha, curr_obs = \
            simulation_ppd[site.mu], \
            simulation_ppd[site.beta], \
            simulation_ppd[site.alpha], \
            simulation_ppd[site.obs]

        if i == 0:
            mu, beta, alpha, obs = curr_mu, curr_beta, curr_alpha, curr_obs
        else:
            mu = np.concatenate([mu, curr_mu], axis=1)
            beta = np.concatenate([beta, curr_beta], axis=1)
            alpha = np.concatenate([alpha, curr_alpha], axis=1)
            obs = np.concatenate([obs, curr_obs], axis=1)

        simulation_ppd = None
        curr_mu, curr_beta, curr_alpha, curr_obs = None, None, None, None
        del simulation_ppd, curr_mu, curr_beta, curr_alpha, curr_obs
        gc.collect()

    simulation_df = \
        pd.concat([simulation_df] * TOTAL_REPS, ignore_index=True) \
        .reset_index(drop=True) \
        .copy()

    logger.info(f"Combined dataframe: {simulation_df.shape}")

    simulation_ppd = simulation_params.copy()
    simulation_ppd[site.mu] = mu
    simulation_ppd[site.beta] = beta
    simulation_ppd[site.alpha] = alpha
    simulation_ppd[site.obs] = obs

    logger.info(mu.shape)

    assert simulation_df.shape[0] == simulation_ppd[site.mu].shape[1]
    assert simulation_df.shape[0] == simulation_ppd[site.beta].shape[1]
    assert simulation_df.shape[0] == simulation_ppd[site.alpha].shape[1]
    assert simulation_df.shape[0] == simulation_ppd[site.obs].shape[1]

    logger.info(f"simulation_ppd:")
    for u, v in simulation_ppd.items():
        logger.info(f"{u}: {v.shape}")

    """ Save simulation dataframe and posterior predictive """
    dest = os.path.join(simulator.build_dir, "simulation_df.csv")
    simulation_df.to_csv(dest, index=False)
    logger.info(f"Saved simulation dataframe to {dest}")

    dest = os.path.join(simulator.build_dir, "simulation_ppd.pkl")
    with open(dest, "wb") as f:
        pickle.dump((simulator, simulation_ppd), f)
    logger.info(f"Saved simulation posterior predictive to {dest}")

    return simulation_df, simulation_params


def main():
    """ Load simulated dataframe """
    src = SIMULATION_DF
    simulation_df = pd.read_csv(src)

    """ Load simulated parameters """
    src = SIMULATION_PARAMS
    with open(src, "rb") as f:
        simulator, simulation_params = pickle.load(f)

    setup_logging(
        dir=simulator.build_dir,
        fname=os.path.basename(__file__)
    )

    combine(simulator, simulation_df, simulation_params)
    return

if __name__ == "__main__":
    main()
