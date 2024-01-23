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
from models import HierarchicalBayesianModel, NonHierarchicalBayesianModel, MaximumLikelihoodModel
from utils import generate_nested_pulses
from constants import (TOML_PATH, REP)

logger = logging.getLogger(__name__)

SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data"
SIMULATION_DF_PATH = os.path.join(SIMULATION_DIR, "simulation_df.csv")
SIMULATION_PPD_PATH = os.path.join(SIMULATION_DIR, "simulation_ppd.pkl")

EXPERIMENT_NAME = "number_of_subjects"
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
        n_reps,
        n_pulses,
        n_subjects,
        draw,
        M
    ):
        # Artefacts directory
        n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
        draw_dir = f"d{draw}"

        # Load data
        if M.NAME in ["hbm"]:
            ind = \
                (simulation_df[simulator.features[0]] < n_subjects) & \
                (simulation_df[REP] < n_reps) & \
                (simulation_df[simulator.intensity].isin(pulses_map[n_pulses]))
            df = simulation_df[ind].reset_index(drop=True).copy()
            df[simulator.response[0]] = ppd_obs[draw, ind, 0]

            ind = df[simulator.response[0]] > 0
            df = df[ind].reset_index(drop=True).copy()

            # Build model
            toml_path = TOML_PATH
            config = Config(toml_path=toml_path)
            config.BUILD_DIR = os.path.join(
                experiment_dir,
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
            posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
            model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

            # Compute error and save results
            a_true = ppd_a[draw, :n_subjects, ...]
            a_pred = posterior_samples[site.a]
            assert a_pred.mean(axis=0).shape == a_true.shape
            np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
            np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

            config, df, prediction_df, encoder_dict, _,  = None, None, None, None, None
            model, posterior_samples, posterior_predictive = None, None, None
            a_true, a_pred = None, None
            del config, df, prediction_df, encoder_dict, _
            del model, posterior_samples, posterior_predictive
            del a_true, a_pred
            gc.collect()

        # Non-hierarchical Bayesian model needs to be run separately on individual subjects
        # otherwise, there are convergence issues when the number of subjects is large
        elif M.NAME in ["nhbm", "mle"]:
            for subject in range(n_subjects):
                sub_dir = f"subject{subject}"

                ind = \
                    (simulation_df[simulator.features[0]] == subject) & \
                    (simulation_df[REP] < n_reps) & \
                    (simulation_df[simulator.intensity].isin(pulses_map[n_pulses]))
                df = simulation_df[ind].reset_index(drop=True).copy()
                df[simulator.response[0]] = ppd_obs[draw, ind, 0]

                ind = df[simulator.response[0]] > 0
                df = df[ind].reset_index(drop=True).copy()

                # Build model
                toml_path = TOML_PATH
                config = Config(toml_path=toml_path)
                config.BUILD_DIR = os.path.join(
                    experiment_dir,
                    draw_dir,
                    n_subjects_dir,
                    n_reps_dir,
                    n_pulses_dir,
                    M.NAME,
                    sub_dir
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
                posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
                model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

                # Compute error and save results
                a_true = ppd_a[draw, [subject], ...]
                a_pred = posterior_samples[site.a]
                assert a_pred.mean(axis=0).shape == a_true.shape
                np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
                np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

                config, df, prediction_df, encoder_dict, _,  = None, None, None, None, None
                model, posterior_samples, posterior_predictive = None, None, None
                a_true, a_pred = None, None
                del config, df, prediction_df, encoder_dict, _
                del model, posterior_samples, posterior_predictive
                del a_true, a_pred
                gc.collect()
        return


    draws_space = np.arange(ppd_obs.shape[0])

    # # Run for Hierarchical Bayesian Model
    # models = [HierarchicalBayesianModel]

    # Run for Non-hierarchical Bayesian Model
    # n_subjects_space = [16]
    # models = [NonHierarchicalBayesianModel]

    # Run for Non-hierarchical Bayesian Model
    n_subjects_space = [16]
    # models = [MaximumLikelihoodModel]
    models = [MaximumLikelihoodModelRecLog]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(
                N_REPS, N_PULSES, n_subjects, draw, M
            ) \
            for draw in draws_space \
            for n_subjects in n_subjects_space \
            for M in models
        )


if __name__ == "__main__":
    main()
