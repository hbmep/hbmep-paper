import os
import pickle
import logging

import pandas as pd
import numpy as np
from jax import random

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models__paired import LearnPosteriorModel
from models__group import Simulator
from utils import generate_group_simulation_dirs
from constants__paired import (
    TOML_PATH,
    LEARN_POSTERIOR_DIR,
    REP,
    INFERENCE_FILE,
    SIMULATION_DF,
    TOTAL_SUBJECTS,
    TOTAL_PULSES,
)

logger = logging.getLogger(__name__)

POSTERIOR_PATH = os.path.join(LEARN_POSTERIOR_DIR, INFERENCE_FILE)
MIN_VALID_DRAWS = 2000


def main(a_loc_delta, build_dir):
    # Build simulator
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = build_dir
    simulator = Simulator(
        config=config, a_loc_delta=a_loc_delta
    )

    # Set up logging
    os.makedirs(build_dir, exist_ok=True)
    setup_logging(
        dir=simulator.build_dir,
        fname=os.path.basename(__file__)
    )

    logger.info(f"a_loc_delta: {a_loc_delta}")
    logger.info(f"build_dir: {build_dir}")

    # Create template dataframe for simulation
    simulation_df = (
        pd.DataFrame(
            np.arange(0, TOTAL_SUBJECTS, 1),
            columns=[simulator.features[0]]
        )
        .merge(
            pd.DataFrame(
                np.arange(0, 2, 1),
                columns=[simulator.features[1]]
            ),
            how="cross"
        )
    )
    simulation_df[simulator.intensity] = 0
    simulation_df = simulator.make_prediction_dataset(
        df=simulation_df,
        min_intensity=0,
        max_intensity=100,
        num_points=TOTAL_PULSES
    )
    simulation_df[REP] = 0
    logger.info(f"Simulation dataframe: {simulation_df.shape}")

    # Load learnt posterior
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        _, _, posterior_samples = pickle.load(g)

    logger.info("Learn posterior shapes:")
    for u, v in posterior_samples.items():
        logger.info(f"{u}: {v.shape}")

    # a_loc (4000,)      -> a_loc_fixed (4000, 1, 4)

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

    # Rename sites
    posterior_samples["a_loc_fixed"] = posterior_samples.pop("a_loc")
    posterior_samples["a_loc_fixed"] = posterior_samples["a_loc_fixed"][:, None, None]
    posterior_samples["a_loc_fixed"] = np.repeat(posterior_samples["a_loc_fixed"], repeats=4, axis=-1)
    logger.info(f"Renamed sites: {sorted(list(posterior_samples.keys()))}")

    # Simulate
    logger.info(f"Simulating new subjects ...")
    simulation_ppd = \
        simulator.predict(
            df=simulation_df,
            posterior_samples=posterior_samples
        )
    logger.info(f"simulation_ppd: {sorted(list(simulation_ppd.keys()))}")

    for u, v in simulation_ppd.items():
        logger.info(f"{u}: {v.shape}")

    # Ensure positive observations
    assert (simulation_ppd[site.obs] > 0).all()

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
    simulation_data_dirs = generate_group_simulation_dirs()

    key = "with_effect"
    a_loc_delta, build_dir = simulation_data_dirs[key]
    main(a_loc_delta, build_dir)

    key = "no_effect"
    a_loc_delta, build_dir = simulation_data_dirs[key]
    main(a_loc_delta, build_dir)
