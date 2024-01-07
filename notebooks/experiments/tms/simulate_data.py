import functools
import os
import pickle
import logging

import pandas as pd
import numpy as np

from hbmep.config import Config
from hbmep.model.utils import Site as site

from models import LearnPosterior, Simulator

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

TOTAL_SUBJECTS = 1000
TOTAL_PULSES = 60
MIN_VALID_SUBJECTS_PER_DRAW = 200


def main():
    a_random_mean, a_random_scale = -2.5, 1.5
    toml_path = "/home/vishu/repos/hbmep-paper/configs/experiments/subjects.toml"
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "simulate-data", f"a_random_mean_{a_random_mean}_a_random_scale_{a_random_scale}")

    simulator = Simulator(config=config, a_random_mean=a_random_mean, a_random_scale=a_random_scale)
    simulator._make_dir(simulator.build_dir)
    dest = os.path.join(simulator.build_dir, "log.log")
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

    """ Load learn posterior """
    src = "/home/vishu/repos/hbmep-paper/reports/experiments/subjects/learn-posterior/inference.pkl"
    with open(src, "rb") as g:
        _, _, posterior_samples_learnt = pickle.load(g)

    for k, v in posterior_samples_learnt.items():
        logger.info(f"{k}: {v.shape}")

    """ Create template dataframe for simulation """
    simulation_df = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[simulator.features[0]]) \
        .merge(
            pd.DataFrame(np.arange(0, 2, 1), columns=[simulator.features[1]]),
            how="cross"
        ) \
        .merge(
            pd.DataFrame([0, 90], columns=[simulator.intensity]),
            how="cross"
        )
    simulation_df = simulator.make_prediction_dataset(
        df=simulation_df,
        min_intensity=0,
        max_intensity=100,
        num=TOTAL_PULSES
    )
    logger.info(f"Simulation dataframe: {simulation_df.shape}")

    """ Simulate data """
    priors = {
        "global-priors": [
            "b_scale_global_scale",
            "v_scale_global_scale",
            "L_scale_global_scale",
            "ell_scale_global_scale",
            "H_scale_global_scale",
            "c_1_scale_global_scale",
            "c_2_scale_global_scale"
        ],
        "hyper-priors": [
            "b_scale",
            "v_scale",
            "L_scale",
            "ell_scale",
            "H_scale",
            "c_1_scale",
            "c_2_scale"
        ],
        "baseline-priors": [
            "a_fixed_mean",
            "a_fixed_scale"
        ]
    }
    sites_to_use_for_simulation = priors["global-priors"]
    sites_to_use_for_simulation += priors["hyper-priors"]
    sites_to_use_for_simulation += priors["baseline-priors"]
    logger.info(f"sites_to_use_for_simulation: {', '.join(sites_to_use_for_simulation)}")
    posterior_samples_learnt = {
        k: v for k, v in posterior_samples_learnt.items() if k in sites_to_use_for_simulation
    }
    logger.info(f"Simulating ...")
    simulation_ppd = \
        simulator.predict(df=simulation_df, posterior_samples=posterior_samples_learnt)

    """ Valid draws """
    a = simulation_ppd[site.a]
    b = simulation_ppd[site.b]
    H = simulation_ppd[site.H]
    logger.info(f"a: {a.shape}")
    logger.info(f"b: {b.shape}")
    logger.info(f"H: {H.shape}")

    filter = (a > 20) & (a < 70) & (b > .05) & (H > .1)
    filter = filter.all(axis=(-1, -2))
    min_valid_subjects_per_draw = filter.sum(axis=-1).min()
    logger.info(f"Filter shape: {filter.shape}")
    logger.info(f"Min valid subjects per draw: {min_valid_subjects_per_draw}")
    assert min_valid_subjects_per_draw >= MIN_VALID_SUBJECTS_PER_DRAW

    dest = os.path.join(simulator.build_dir, "filter.npy")
    np.save(dest, filter)
    logger.info(f"Saved filter to {dest}")

    # filter = ((a > 0) & (a < 100)).all(axis=(-1, -2, -3))
    # simulation_ppd = \
    #     {
    #         k: v[filter, ...] for k, v in simulation_ppd.items()
    #     }

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
