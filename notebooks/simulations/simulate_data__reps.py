import os
import pickle
import logging

import pandas as pd
import numpy as np
from jax import random

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models__accuracy import HierarchicalBayesianModel
from models__power import Simulator
from utils import generate_reps_map
from constants__accuracy import (
    TOML_PATH as TOML_PATH_ACCURACY,
    LEARN_POSTERIOR_DIR,
    REP,
    INFERENCE_FILE,
    SIMULATION_DF
)
from constants__power import (
    TOML_PATH as TOML_PATH_POWER
)
from constants__reps import (
    REPS_DIR__ACCURACY,
    REPS_DIR__POWER,
    N_SUBJECTS,
    TOTAL_SUBJECTS,
    TOTAL_PULSES,
    N_SUBJECTS_SPACE,
    N_PULSES_SPACE,
    N_REPS_SPACE
)

logger = logging.getLogger(__name__)
POSTERIOR_PATH = os.path.join(LEARN_POSTERIOR_DIR, INFERENCE_FILE)
MIN_VALID_DRAWS = 2000


def simulate_data_accuracy():
    build_dir = REPS_DIR__ACCURACY

    # Build simulator
    config = Config(toml_path=TOML_PATH_ACCURACY)
    config.BUILD_DIR = build_dir
    simulator = HierarchicalBayesianModel(config=config, simulate=True)

    # Set up logging
    os.makedirs(simulator.build_dir, exist_ok=True)
    setup_logging(
        dir=simulator.build_dir,
        fname=os.path.basename(__file__)
    )

    # Create template dataframe for simulation
    pulses, reps_map = generate_reps_map(
        n_reps_space=N_REPS_SPACE,
        n_pulses_space=N_PULSES_SPACE
    )
    simulation_df = pd.DataFrame(
        np.arange(0, N_SUBJECTS, 1),
        columns=[simulator.features[0]]
    )
    simulation_df[simulator.intensity] = (
        simulation_df[simulator.features[0]]
        .apply(lambda _: np.array(pulses))
    )
    simulation_df = simulation_df.explode(column=simulator.intensity)[simulator.regressors].copy()
    simulation_df[simulator.intensity] = simulation_df[simulator.intensity].astype(float)
    simulation_df = simulation_df.reset_index(drop=True).copy()

    intensity = simulation_df[simulator.intensity].values.tolist()
    for u, v in reps_map.items():
        assert set(v) <= set(intensity)

    simulation_df = (
        pd.concat([simulation_df] * N_REPS_SPACE[-1], ignore_index=True)
        .reset_index(drop=True)
        .copy()
    )
    arr = []
    for i in range(0, N_REPS_SPACE[-1]):
        arr += [i] * N_SUBJECTS * len(pulses)
    simulation_df[REP] = arr
    logger.info(f"Simulation dataframe: {simulation_df.shape}")

    # Load learnt posterior
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        _, _, posterior_samples = pickle.load(g)

    logger.info("Learn posterior shapes:")
    for u, v in posterior_samples.items():
        logger.info(f"{u}: {v.shape}")

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


def simulate_data_power(a_delta_loc, a_delta_scale):
    build_dir = REPS_DIR__POWER

    # Build simulator
    config = Config(toml_path=TOML_PATH_POWER)
    config.BUILD_DIR = build_dir
    simulator = Simulator(
        config=config, a_delta_loc=a_delta_loc, a_delta_scale=a_delta_scale
    )

    # Set up logging
    os.makedirs(simulator.build_dir, exist_ok=True)
    setup_logging(
        dir=simulator.build_dir,
        fname=os.path.basename(__file__)
    )

    # Create template dataframe for simulation
    pulses, reps_map = generate_reps_map(
        n_reps_space=N_REPS_SPACE,
        n_pulses_space=[TOTAL_PULSES]
    )
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
    simulation_df[simulator.intensity] = (
        simulation_df[simulator.features[0]]
        .apply(lambda _: np.array(pulses))
    )
    simulation_df = simulation_df.explode(column=simulator.intensity)[simulator.regressors].copy()
    simulation_df[simulator.intensity] = simulation_df[simulator.intensity].astype(float)
    simulation_df = simulation_df.reset_index(drop=True).copy()

    intensity = simulation_df[simulator.intensity].values.tolist()
    for u, v in reps_map.items():
        assert set(v) <= set(intensity)

    simulation_df = (
        pd.concat([simulation_df] * N_REPS_SPACE[-1], ignore_index=True)
        .reset_index(drop=True)
        .copy()
    )
    arr = []
    for i in range(0, N_REPS_SPACE[-1]):
        arr += [i] * TOTAL_SUBJECTS * len(pulses) * 2
    simulation_df[REP] = arr
    logger.info(f"Simulation dataframe: {simulation_df.shape}")

    # Load learnt posterior
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        _, _, posterior_samples = pickle.load(g)

    logger.info("Learn posterior shapes:")
    for u, v in posterior_samples.items():
        logger.info(f"{u}: {v.shape}")

    # a_loc (4000,)      -> a_fixed_loc (4000,)
    # a_scale (4000,)    -> a_fixed_scale (4000,)
    # ... generates a_fixed (4000, n_subjects, 1, 1)
    # ... generates a_delta (4000, n_subjects, 1, 1)

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
    posterior_samples["a_fixed_loc"] = posterior_samples.pop("a_loc")
    posterior_samples["a_fixed_scale"] = posterior_samples.pop("a_scale")
    logger.info(f"Renamed sites: {sorted(list(posterior_samples.keys()))}")

    # Simulate
    logger.info(f"Simulating new subjects ...")
    simulation_ppd = \
        simulator.predict(
            df=simulation_df,
            posterior_samples=posterior_samples
        )
    logger.info(f"simulation_ppd: {sorted(list(simulation_ppd.keys()))}")

    # Exclude invalid draws based on negative thresholds
    flag_valid_draws = (simulation_ppd[site.a] > 0).all(axis=(1, 2, 3))
    assert flag_valid_draws.sum() > MIN_VALID_DRAWS
    logger.info(f"Valid draws: {flag_valid_draws.mean() * 100:.2f}%")
    for u, v in simulation_ppd.items():
        simulation_ppd[u] = v[flag_valid_draws, ...]

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
    # # Accuracy
    # simulate_data_accuracy()
    # Power
    simulate_data_power(-5., 2.5)
