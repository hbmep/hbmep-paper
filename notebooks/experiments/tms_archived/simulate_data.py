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
from hbmep_paper.utils import setup_logging
from utils import fix_keys_for_reps
from constants import (
    TOTAL_SUBJECTS,
    TOTAL_PULSES,
    A_RANDOM_MEAN,
    A_RANDOM_SCALE,
    TOTAL_REPS,
    MIN_VALID_SUBJECTS_PER_DRAW
)

logger = logging.getLogger(__name__)

POSTERIOR_PATH = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/learn_posterior/inference.pkl"
BUILD_DIR = f"/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_{A_RANDOM_MEAN}_a_random_scale_{A_RANDOM_SCALE}/"


def simulate_params(simulator):
    """ Load learnt posterior """
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        _, _, posterior_samples = pickle.load(g)

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

    """ Exclude priors """
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

    """ Simulate """
    logger.info(f"Simulating parameters for new subjects ...")
    simulation_ppd = \
        simulator.predict(
            df=simulation_df,
            posterior_samples=posterior_samples
        )
    logger.info(f"simulation_ppd: {sorted(list(simulation_ppd.keys()))}")
    simulation_params = {
        u: v for u, v in simulation_ppd.items() if u not in [site.mu, site.beta, site.alpha, site.obs]
    }   # Keep only the parameters
    for u, v in posterior_samples.items():
        assert u not in simulation_params.keys()
        simulation_params[u] = v
    logger.info(f"simulation_params: {sorted(list(simulation_params.keys()))}")

    """ Shuffle draws """
    logger.info(f"Shuffling draws ...")
    ind = np.arange(0, simulation_params[site.a].shape[0], 1)
    ind = jax.random.permutation(simulator.rng_key, ind)
    ind = np.array(ind)
    simulation_params = \
        {
            k: v[ind, ...] for k, v in simulation_params.items()
        }

    return simulation_df, simulation_params


def simulate_data(simulator, simulation_df, simulation_params):
    """ Simulate data """
    rng_keys = fix_keys_for_reps(simulator.rng_key)
    assert len(rng_keys) == TOTAL_REPS
    return_sites = [
        site.mu, site.beta, site.alpha, site.obs
    ]

    logger.info(f"Simulation repeated observations using simulated parameters ...")
    for i in range(TOTAL_REPS):     # Reps
        logger.info(f"Simulating for rep {i + 1} / {TOTAL_REPS} ...")
        simulation_ppd = \
            simulator.predict(
                df=simulation_df,
                posterior_samples=simulation_params,
                return_sites=return_sites,
                rng_key=rng_keys[i]
            )

        dest = os.path.join(simulator.build_dir, f"simulation_ppd_{i}.pkl")
        with open(dest, "wb") as f:
            pickle.dump((simulation_ppd,), f)
        logger.info(f"Saved simulated posterior predictive to {dest}")

        simulation_ppd = None
        del simulation_ppd
        gc.collect()

    """ Valid draws """
    a = simulation_params[site.a]
    mask = (a > 0) & (a < 100)
    mask = mask.all(axis=(-1, -2))
    min_valid_subjects_per_draw = mask.sum(axis=-1).min()
    logger.info(f"Mask shape: {mask.shape}")
    logger.info(f"Min. valid subjects per draw: {min_valid_subjects_per_draw}")
    assert min_valid_subjects_per_draw >= MIN_VALID_SUBJECTS_PER_DRAW

    """ Save mask """
    dest = os.path.join(simulator.build_dir, "mask.npy")
    np.save(dest, mask)
    logger.info(f"Saved mask to {dest}")

    """ Save simulation dataframe and parameters """
    logger.info(f"Saving results ...")
    dest = os.path.join(simulator.build_dir, "simulation_df.csv")
    simulation_df.to_csv(dest, index=False)
    logger.info(f"Saved simulation dataframe to {dest}")

    dest = os.path.join(simulator.build_dir, "simulation_params.pkl")
    with open(dest, "wb") as f:
        pickle.dump((simulator, simulation_params), f)
    logger.info(f"Saved simulation parameters to {dest}")
    return


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

    simulation_df, simulation_params = simulate_params(simulator)
    simulate_data(simulator, simulation_df, simulation_params)
    return


if __name__ == "__main__":
    main()
