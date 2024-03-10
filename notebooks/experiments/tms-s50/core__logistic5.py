import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import (
    RectifiedLogistic,
    Logistic5,
    Logistic4
)
from utils import generate_nested_pulses
from constants import (
    TOML_PATH,
    REP,
    SIMULATE_DATA_LOGISTIC5_DIR as SIMULATE_DATA_DIR,
    SIMULATION_DF,
    INFERENCE_FILE,
    EXPERIMENTS_LOGISTIC5_DIR as NUMBER_OF_PULSES_DIR,
    N_PULSES_SPACE
)

logger = logging.getLogger(__name__)

SIMULATION_DF_PATH = os.path.join(SIMULATE_DATA_DIR, SIMULATION_DF)
SIMULATION_PPD_PATH = os.path.join(SIMULATE_DATA_DIR, INFERENCE_FILE)
BUILD_DIR = NUMBER_OF_PULSES_DIR

N_REPS = 1
N_SUBJECTS = 8


@timing
def main():
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

        # Load data
        ind = (
            (simulation_df[simulator.features[0]] < n_subjects) &
            (simulation_df[REP] < n_reps) &
            (simulation_df[simulator.intensity].isin(pulses_map[n_pulses]))
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

        return


    # Experiment space
    draws_space = range(0, 100)
    n_pulses_space = N_PULSES_SPACE
    n_jobs = -1

    models = [
        RectifiedLogistic,
        Logistic5,
        Logistic4
    ]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(
                N_REPS, n_pulses, N_SUBJECTS, draw, M
            )
            for draw in draws_space
            for n_pulses in n_pulses_space
            for M in models
        )


if __name__ == "__main__":
    main()
