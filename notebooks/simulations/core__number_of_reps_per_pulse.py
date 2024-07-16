import os
import sys
import gc
import pickle
import logging

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models__accuracy import HierarchicalBayesianModel
from utils import generate_nested_pulses
from constants__accuracy import (
    TOML_PATH,
    SIMULATE_DATA_DIR__ACCURACY,
    NUMBER_OF_REPS_PER_PULSE_DIR,
    N_PULSES_SPACE,
    N_REPS_PER_PULSE_SPACE,
    REP,
    INFERENCE_FILE,
    SIMULATION_DF,
)

logger = logging.getLogger(__name__)

SIMULATION_DF_PATH = os.path.join(SIMULATE_DATA_DIR__ACCURACY, SIMULATION_DF)
SIMULATION_PPD_PATH = os.path.join(SIMULATE_DATA_DIR__ACCURACY, INFERENCE_FILE)
BUILD_DIR = NUMBER_OF_REPS_PER_PULSE_DIR

N_SUBJECTS = 8
N_PULSES_SPACE = N_PULSES_SPACE[2:]


@timing
def main(draws_space, n_reps_space, n_pulses_space, n_jobs=-1):
    # Load simulated dataframe
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    ppd_a = simulation_ppd[site.a]
    ppd_obs = simulation_ppd[site.obs]
    simulation_ppd = None
    del simulation_ppd
    gc.collect()

    # Set up logging
    simulator._make_dir(BUILD_DIR)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )

    # Generate nested pulses
    pulses_map = generate_nested_pulses(simulator, simulation_df)


    # Define experiment
    def run_experiment(
        n_reps,
        n_pulses,
        n_subjects,
        draw,
        M
    ):
        # Required for build directory
        n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
        draw_dir = f"d{draw}"

        match M.NAME:
            case "hierarchical_bayesian_model":
                # Load data
                pulses = pulses_map[n_pulses][::n_reps]
                ind = (
                    (simulation_df[simulator.features[0]] < n_subjects) &
                    (simulation_df[REP] < n_reps) &
                    (simulation_df[simulator.intensity].isin(pulses))
                )
                df = simulation_df[ind].reset_index(drop=True).copy()
                df[simulator.response[0]] = ppd_obs[draw, ind, 0]

                ind = df[simulator.response[0]] > 0
                df = df[ind].reset_index(drop=True).copy()

                # Build model
                config = Config(toml_path=TOML_PATH)
                config.BUILD_DIR = os.path.join(
                    BUILD_DIR,
                    draw_dir,
                    n_subjects_dir,
                    n_reps_dir,
                    n_pulses_dir,
                    M.NAME
                )
                model = M(config=config)

                # Set up logging
                model._make_dir(model.build_dir)
                setup_logging(
                    dir=model.build_dir,
                    fname="logs"
                )

                # Run inference
                df, encoder_dict = model.load(df=df)
                _, posterior_samples = model.run_inference(df=df)

                # Predictions and recruitment curves
                prediction_df = model.make_prediction_dataset(df=df)
                posterior_predictive = model.predict(
                    df=prediction_df, posterior_samples=posterior_samples
                )
                model.render_recruitment_curves(
                    df=df,
                    encoder_dict=encoder_dict,
                    posterior_samples=posterior_samples,
                    prediction_df=prediction_df,
                    posterior_predictive=posterior_predictive
                )

                # Compute error and save results
                a_true = ppd_a[draw, :n_subjects, ...]
                a_pred = posterior_samples[site.a]
                assert a_pred.mean(axis=0).shape == a_true.shape
                np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
                np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

                config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
                model, posterior_samples, posterior_predictive = None, None, None
                a_true, a_pred = None, None
                del config, df, prediction_df, encoder_dict, _
                del model, posterior_samples, posterior_predictive
                del a_true, a_pred
                gc.collect()

            case _:
                raise ValueError(f"Invalid model {M.NAME}.")

        return


    logger.info("Number of reps per pulse experiment.")
    logger.info(f"n_reps_space: {', '.join(map(str, n_reps_space))}")
    logger.info(f"n_pulses_space: {', '.join(map(str, n_pulses_space))}")
    logger.info(f"Running draws {draws_space.start} to {draws_space.stop - 1}.")
    logger.info(f"n_jobs: {n_jobs}")

    M = HierarchicalBayesianModel

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(
                n_reps, n_pulses, N_SUBJECTS, draw, M
            )
            for draw in draws_space
            for n_reps in n_reps_space
            for n_pulses in n_pulses_space
        )


if __name__ == "__main__":
    # Usage: python -m core__number_of_reps_per_pulse 0 4000
    lo, hi = list(map(int, sys.argv[1:]))

    # Experiment space
    draws_space = range(lo, hi)
    n_jobs = -1
    n_pulses_space = N_PULSES_SPACE
    n_reps_space = N_REPS_PER_PULSE_SPACE

    main(
        draws_space=draws_space,
		n_reps_space=n_reps_space,
		n_pulses_space=n_pulses_space,
		n_jobs=n_jobs
    )
