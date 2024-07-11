import functools
import os
import pickle
import logging

import pandas as pd
import numpy as np

from hbmep.config import Config
from hbmep.model.utils import Site as site

#from models_old import RectifiedLogistic
from hbmep.model import RectifiedLogistic

from learn_posterior import TOML_PATH, DATA_PATH

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Change this to indicate path to inference.pkl from learn_posterior.py
POSTERIOR_PATH = "/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N30_triple_muscle_a/learn_posterior/inference.pkl"


def main():
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "simulate_data")

    simulator = RectifiedLogistic(config=config)
    simulator._make_dir(simulator.build_dir)

    """ Set up logging in build directory """
    dest = os.path.join(simulator.build_dir, "simulate_data.log")
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

    """ Load learnt posterior """
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        model, mcmc, posterior_samples, df_ = pickle.load(g)

    logger.info(
        f"Logging all sites learnt posterior with their shapes ..."
    )   # Useful for freezing priors
    for k, v in posterior_samples.items():
        logger.info(f"{k}: {v.shape}")

    """
    Simulate existing participants
    at the same exact (existing) intensities.
    This serves as a basic visual check.

    Since we're simulating existing participants,
    we don't need to freeze the priors.
    We will freeze the priors when we simulate new participants.
    """
    src = DATA_PATH
    df = pd.read_csv(src)

    ind = df["participant_condition"].isin(["Uninjured"])
    df = df[ind].reset_index(drop=True).copy()

    df, _ = model.load(df=df)
    simulation_df = df.copy()

    logger.info(f"Simulating existing participants ...")
    simulation_ppd = \
        simulator.predict(df=simulation_df, posterior_samples=posterior_samples)

    # I'll explain this bit later, you can ignore it for now
    if site.a not in simulation_ppd:
        simulation_ppd[site.a] = posterior_samples[site.a]

    # Save
    dest = os.path.join(simulator.build_dir, "simulation_df_existing_participants.csv")
    simulation_df.to_csv(dest, index=False)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(simulator.build_dir, "simulation_ppd_existing_participants.pkl")
    with open(dest, "wb") as f:
        pickle.dump((simulator, simulation_ppd), f)
    logger.info(f"Saved to {dest}")

    """
    Simulate existing participants
    at equi-spaced intensities.
    We don't need to freeze the priors.
    """

    src = DATA_PATH
    df = pd.read_csv(src)

    ind = df["participant_condition"].isin(["Uninjured"])
    df = df[ind].reset_index(drop=True).copy()

    df, _ = model.load(df=df)
    TOTAL_PULSES = 60
    simulation_df = simulator.make_prediction_dataset(
        df=df,
        min_intensity=0,
        max_intensity=100,
        num_points=TOTAL_PULSES
    )

    logger.info(f"Simulating existing participants at equi-spaced pulses ...")
    simulation_ppd = \
        simulator.predict(df=simulation_df, posterior_samples=posterior_samples)

    # I'll explain this bit later, you can ignore it for now
    if site.a not in simulation_ppd:
        simulation_ppd[site.a] = posterior_samples[site.a]

    # Save
    dest = os.path.join(simulator.build_dir, "simulation_df_existing_participants_equi_spaced_pulses.csv")
    simulation_df.to_csv(dest, index=False)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(simulator.build_dir, "simulation_ppd_existing_participants_equi_spaced_pulses.pkl")
    with open(dest, "wb") as f:
        pickle.dump((simulator, simulation_ppd), f)
    logger.info(f"Saved to {dest}")


    """
    Simulate new participants
    at equi-spaced intensities.
    We need to freeze the priors.
    """
    # Simulate TOTAL_SUBJECTS subjects
    TOTAL_SUBJECTS = 20
    # Create template dataframe for simulation
    simulation_df = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[simulator.features[0]]) \
        .merge(
            pd.DataFrame([0, 90], columns=[simulator.intensity]),
            how="cross"
        )
    simulation_df = simulator.make_prediction_dataset(
        df=simulation_df,
        min_intensity=0,
        max_intensity=100,
        num_points=TOTAL_PULSES
    )
    logger.info(
        f"Simulation (new participants) dataframe: {simulation_df.shape}"
    )

    # Freeze priors
    sites_to_exclude = {
        site.a, site.b, site.v,
        site.L, site.ell, site.H,
        site.c_1, site.c_2,
        site.mu, site.beta,
        site.obs
    }
    posterior_samples = {
        k: v for k, v in posterior_samples.items() if k not in sites_to_exclude
    }

    logger.info(f"Simulating new participants ...")
    simulation_ppd = \
        simulator.predict(df=simulation_df, posterior_samples=posterior_samples)
    # This will always contain site.a because we're supplying posterior_samples
    # without site.a

    # Save
    dest = os.path.join(simulator.build_dir, "simulation_df_new_participants.csv")
    simulation_df.to_csv(dest, index=False)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(simulator.build_dir, "simulation_ppd_new_participants.pkl")
    with open(dest, "wb") as f:
        pickle.dump((simulator, simulation_ppd), f)
    logger.info(f"Saved to {dest}")

    return


if __name__ == "__main__":
    main()
