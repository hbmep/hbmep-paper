import os
import pickle
import logging

import pandas as pd
import numpy as np
from jax import random
import jax.numpy as jnp

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from constants__accuracy import LEARN_POSTERIOR_DIR
from models__accuracy import HierarchicalBayesianModel
from constants__saturation import (
    TOML_PATH,
    SIMULATE_DATA_DIR__SATURATION,
    TOTAL_SUBJECTS,
    TOTAL_PULSES,
    TOTAL_REPS,
    MAX_INTENSITY,
    REP,
    INFERENCE_FILE,
    SIMULATION_DF
)

logger = logging.getLogger(__name__)

POSTERIOR_PATH = os.path.join(LEARN_POSTERIOR_DIR, INFERENCE_FILE)
BUILD_DIR = SIMULATE_DATA_DIR__SATURATION


def main():
    # Build simulator
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = BUILD_DIR
    simulator = HierarchicalBayesianModel(config=config)

    # Set up logging
    simulator._make_dir(simulator.build_dir)
    setup_logging(
        dir=simulator.build_dir,
        fname=os.path.basename(__file__)
    )

    # Create template dataframe for simulation
    simulation_df = pd.DataFrame(
        np.arange(0, TOTAL_SUBJECTS, 1),
        columns=[simulator.features[0]]
    )
    simulation_df[simulator.intensity] = 0
    simulation_df = simulator.make_prediction_dataset(
        df=simulation_df,
        min_intensity=0,
        max_intensity=MAX_INTENSITY,
        num_points=TOTAL_PULSES
    )
    simulation_df = (
        pd.concat([simulation_df] * TOTAL_REPS, ignore_index=True)
        .reset_index(drop=True)
        .copy()
    )
    arr = []
    for i in range(0, TOTAL_REPS):
        arr += [i] * TOTAL_SUBJECTS * TOTAL_PULSES
    simulation_df[REP] = arr
    logger.info(f"Simulation dataframe: {simulation_df.shape}")

    # Load learnt posterior
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        _, _, posterior_samples = pickle.load(g)

    # Exclude priors
    present_sites = sorted(list(posterior_samples.keys()))
    sites_to_exclude = [
        site.a, site.b,
        site.L, site.ell, site.H,
        site.c_1, site.c_2,
        site.mu, site.beta, site.alpha, site.obs
    ]
    sites_to_exclude = sorted(sites_to_exclude)
    logger.info(f"Existing posterior sites: {present_sites}")
    logger.info(f"Sites to exclude: {sites_to_exclude}")
    posterior_samples = {
        u: v for u, v in posterior_samples.items()
        if u not in sites_to_exclude
    }
    remaining_sites = sorted(list(posterior_samples.keys()))
    logger.info(f"Remaining sites: {remaining_sites}")

    # Simulate
    logger.info(f"Simulating new subjects ...")
    simulation_ppd = \
        simulator.predict(
            df=simulation_df,
            posterior_samples=posterior_samples
        )
    logger.info(f"simulation_ppd: {sorted(list(simulation_ppd.keys()))}")

    # Calculate S50
    logger.info(f"Calculating S50 ...")
    named_params = [site.a, site.b, site.L, site.ell, site.H]
    var_L = simulation_ppd[site.L]
    var_H = simulation_ppd[site.H]
    y_at_S50 = var_L + jnp.true_divide(var_H, 2)
    s50 = F.solve_rectified_logistic(
        y_at_S50,
        *[simulation_ppd[u] for u in named_params]
    )
    simulation_ppd[site.s50] = s50

    # Shuffle draws
    logger.info(f"Shuffling draws ...")
    ind = np.arange(0, simulation_ppd[site.a].shape[0], 1)
    _, rng_key = random.split(simulator.rng_key)
    ind = random.permutation(rng_key, ind)
    ind = np.array(ind)
    simulation_ppd = {
        u: v[ind, ...] for u, v in simulation_ppd.items()
    }

    # Save simulation dataframe and posterior predictive
    dest = os.path.join(simulator.build_dir, SIMULATION_DF)
    simulation_df.to_csv(dest, index=False)
    logger.info(f"Saved simulation dataframe to {dest}")

    dest = os.path.join(simulator.build_dir, INFERENCE_FILE)
    with open(dest, "wb") as f:
        pickle.dump((simulator, simulation_ppd), f)
    logger.info(f"Saved simulation posterior predictive to {dest}")
    return


if __name__ == "__main__":
    main()
