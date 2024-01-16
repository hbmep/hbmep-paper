import functools
import os
import pickle
import logging

import pandas as pd
import numpy as np
import jax

from hbmep.config import Config
from hbmep.model.utils import Site as site

from models import LearnPosterior, Simulator
from learn_posterior import TOML_PATH
from utils import setup_logging

logger = logging.getLogger(__name__)


TOTAL_SUBJECTS = 1000
TOTAL_PULSES = 60
MIN_VALID_SUBJECTS_PER_DRAW = 200

A_RANDOM_MEAN, A_RANDOM_SCALE = -2.5, 1.5
BUILD_DIR = f"/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/a_random_mean_{A_RANDOM_MEAN}_a_random_scale_{A_RANDOM_SCALE}"

POSTERIOR_PATH = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/learn_posterior/inference.pkl"


def main():
    a_random_mean, a_random_scale = A_RANDOM_MEAN, A_RANDOM_SCALE
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = BUILD_DIR

    simulator = Simulator(
        config=config,
        a_random_mean=a_random_mean,
        a_random_scale=a_random_scale
    )

    simulator._make_dir(simulator.build_dir)
    setup_logging(
        dir=simulator.build_dir,
        fname=os.path.basename(__file__)
    )

    """ Load learnt posterior """
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        _, _, posterior_samples = pickle.load(g)

    for k, v in posterior_samples.items():
        logger.info(f"{k}: {v.shape}")

    """ Create template dataframe for simulation """
    simulation_df = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[simulator.features[0]]) \
        .merge(
            pd.DataFrame(np.arange(0, 2, 1), columns=[simulator.features[1]]),
            how="cross"
        )
    simulation_df[simulator.intensity] = 0
    simulation_df = simulator.make_prediction_dataset(
        df=simulation_df,
        min_intensity=0,
        max_intensity=100,
        num=TOTAL_PULSES
    )
    logger.info(f"Simulation dataframe: {simulation_df.shape}")

    """ Simulate data """
    present_sites = sorted(list(posterior_samples.keys()))
    sites_to_exclude = [
        "a_fixed", site.a, site.b, site.v,
        site.L, site.ell, site.H,
        site.c_1, site.c_2,
        site.mu, site.beta, site.alpha,
        site.obs
    ]
    sites_to_exclude = sorted(sites_to_exclude)
    remaining_sites = set(present_sites) - set(sites_to_exclude)
    remaining_sites = sorted(list(remaining_sites))
    logger.info(f"Existing posterior sites: {present_sites}")
    logger.info(f"Sites to exclude: {sites_to_exclude}")
    logger.info(f"Remaining sites: {remaining_sites}")
    posterior_samples = {
        k: v for k, v in posterior_samples.items() \
        if k in remaining_sites
    }
    logger.info(f"Simulating ...")
    simulation_ppd = \
        simulator.predict(
            df=simulation_df,
            posterior_samples=posterior_samples
        )

    """ Shuffle draws """
    logger.info(f"Shuffling draws ...")
    ind = np.arange(0, simulation_ppd[site.a].shape[0], 1)
    ind = jax.random.permutation(simulator.rng_key, ind)
    ind = np.array(ind)
    simulation_ppd = \
        {
            k: v[ind, ...] for k, v in simulation_ppd.items()
        }

    """ Valid draws """
    a = simulation_ppd[site.a]
    b = simulation_ppd[site.b]
    H = simulation_ppd[site.H]
    logger.info(f"a: {a.shape}")
    logger.info(f"b: {b.shape}")
    logger.info(f"H: {H.shape}")

    # filter = (a > 20) & (a < 70) & (b > .05) & (H > .1)
    filter = (a > 0) & (a < 100)
    filter = filter.all(axis=(-1, -2))
    min_valid_subjects_per_draw = filter.sum(axis=-1).min()
    logger.info(f"Filter shape: {filter.shape}")
    logger.info(f"Min. valid subjects per draw: {min_valid_subjects_per_draw}")
    assert min_valid_subjects_per_draw >= MIN_VALID_SUBJECTS_PER_DRAW

    """ Save filter """
    dest = os.path.join(simulator.build_dir, "filter.npy")
    np.save(dest, filter)
    logger.info(f"Saved filter to {dest}")

    """ Save simulation dataframe and posterior predictive """
    dest = os.path.join(simulator.build_dir, "simulation_df.csv")
    simulation_df.to_csv(dest, index=False)
    logger.info(f"Saved simulation dataframe to {dest}")

    dest = os.path.join(simulator.build_dir, "simulation_ppd.pkl")
    with open(dest, "wb") as f:
        pickle.dump((simulator, simulation_ppd), f)
    logger.info(f"Saved simulation posterior predictive to {dest}")


if __name__ == "__main__":
    main()
