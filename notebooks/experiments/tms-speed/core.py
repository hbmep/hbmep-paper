import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np
import jax
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import ReLU
from utils import generate_nested_pulses
from constants import (TOML_PATH, REP)

logger = logging.getLogger(__name__)

SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-speed/simulate_data"
SIMULATION_DF_PATH = os.path.join(SIMULATION_DIR, "simulation_df.csv")
SIMULATION_PPD_PATH = os.path.join(SIMULATION_DIR, "simulation_ppd.pkl")

EXPERIMENT_NAME = "speed-relu"
N_PULSES = 48
N_REPS = 1


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

    experiment_dir = os.path.join(SIMULATION_DIR, EXPERIMENT_NAME)
    simulator._make_dir(experiment_dir)
    setup_logging(
        dir=experiment_dir,
        fname=os.path.basename(__file__)
    )

    # Generate nested pulses
    pulses_map = generate_nested_pulses(simulator, simulation_df)

    # Experiment space
    n_subjects_space = [1, 4, 8, 16]
    n_jobs = -1


    # Define experiment
    def run_experiment(
        method,
        draw,
        M,
        n_reps=1,
        n_pulses=48,
        n_subjects=1,

    ):
        # Artefacts directory
        n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
        draw_dir = f"d{draw}"

        ind = \
            (simulation_df[simulator.features[0]] < n_subjects) & \
            (simulation_df[REP] < n_reps) & \
            (simulation_df[simulator.intensity].isin(pulses_map[n_pulses]))
        df = simulation_df[ind].reset_index(drop=True).copy()
        df[simulator.response[0]] = ppd_obs[draw, ind, 0]

        ind = df[simulator.response[0]] > 0
        df = df[ind].reset_index(drop=True).copy()
        logger.info(f"Number of observations: {df.shape[0]}")

        # Build model
        toml_path = TOML_PATH
        config = Config(toml_path=toml_path)
        config.BUILD_DIR = os.path.join(
            experiment_dir,
            draw_dir,
            M.NAME,
            method
        )
        model = M(config=config)

        # Set up logging
        model._make_dir(model.build_dir)
        setup_logging(
            dir=model.build_dir,
            fname="logs"
        )

        df, encoder_dict = model.load(df=df)

        # Run inference
        match method:
            case "mcmc":
                _, posterior_samples, time_taken = model.run_inference(df=df)
            case "svi":
                _, posterior_samples, time_taken = model.run_svi(df=df)
            case "svi_jit":
                losses, posterior_samples, time_taken = model.run_svi_jit(df=df)
                import matplotlib.pyplot as plt
                import seaborn as sns
                sns.lineplot(x=range(len(losses)), y=np.array(losses))
                dest = os.path.join(model.build_dir, "losses.png")
                plt.savefig(dest)
                logger.info(f"Saved to {dest}")
            case "svi_jit_scan":
                losses, posterior_samples, time_taken = model.run_svi_jit_scan(df=df)
            case _:
                raise ValueError(f"Invalid method: {method}")


        # Predictions and recruitment curves
        prediction_df = model.make_prediction_dataset(df=df)
        posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
        model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

        # Compute error and save results
        a_true = ppd_a[draw, :n_subjects, ...]
        a_pred = posterior_samples[site.a]
        assert a_pred.mean(axis=0).shape == a_true.shape
        np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
        np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)
        np.save(os.path.join(model.build_dir, "time_taken.npy"), np.array([time_taken]))
        logger.info(f"Time taken: {time_taken} seconds")
        logger.info(f"MAE: {np.abs(a_true.mean() - a_pred.mean(axis=0)).mean()}")

        config, df, prediction_df, encoder_dict, _,  = None, None, None, None, None
        model, posterior_samples, posterior_predictive = None, None, None
        a_true, a_pred = None, None
        del config, df, prediction_df, encoder_dict, _
        del model, posterior_samples, posterior_predictive
        del a_true, a_pred
        gc.collect()
        return


    draws_space = np.arange(ppd_obs.shape[0])
    # draws_space = draws_space[:5]
    methods_space = ["mcmc", "svi_jit"]
    for draw in draws_space:
        for method in methods_space:
            run_experiment(method, draw, ReLU)

    # run_experiment("mcmc", 0, ReLU)
    # # run_experiment("svi", 0, ReLU)
    # run_experiment("svi_jit", 0, ReLU)
    # # run_experiment("svi_jit_scan", 0, ReLU)

    # with Parallel(n_jobs=n_jobs) as parallel:
    #     parallel(
    #         delayed(run_experiment)(
    #             method, draw, ReLU
    #         )
    #         for draw in draws_space
    #         for method in methods_space
    #     )


if __name__ == "__main__":
    main()
