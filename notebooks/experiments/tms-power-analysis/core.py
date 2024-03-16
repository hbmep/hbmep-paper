import os
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
from models import (
    Simulator,
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel
)
from constants import (
    TOML_PATH,
    REP,
    SIMULATION_DF,
    INFERENCE_FILE,
    TOTAL_REPS,
    TOTAL_PULSES,
    SIMULATE_DATA_DIR,
    SIMULATE_DATA_NO_EFFECT_DIR,
    EXPERIMENTS_DIR,
    EXPERIMENTS_NO_EFFECT_DIR
)

logger = logging.getLogger(__name__)

N_REPS = TOTAL_REPS
N_PULSES = TOTAL_PULSES
N_SUBJECTS_SPACE = [1, 2, 4, 8, 12, 16, 20, 24]


@timing
def main(simulation_data_dir, build_dir):
    # Load simulated dataframe
    src = os.path.join(simulation_data_dir, SIMULATION_DF)
    simulation_df = pd.read_csv(src)

    # Load simulation ppd
    src = os.path.join(simulation_data_dir, INFERENCE_FILE)
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    ppd_a = simulation_ppd[site.a]
    ppd_obs = simulation_ppd[site.obs]
    simulation_ppd = None
    del simulation_ppd
    gc.collect()

    # Set up logging
    simulator._make_dir(build_dir)
    setup_logging(
        dir=build_dir,
        fname=os.path.basename(__file__)
    )


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
                ind = (
                    (simulation_df[simulator.features[0]] < n_subjects) &
                    (simulation_df[REP] < n_reps)
                )
                df = simulation_df[ind].reset_index(drop=True).copy()
                df[simulator.response[0]] = ppd_obs[draw, ind, 0]

                ind = df[simulator.response[0]] > 0
                df = df[ind].reset_index(drop=True).copy()

                # Build model
                config = Config(toml_path=TOML_PATH)
                config.BUILD_DIR = os.path.join(
                    build_dir,
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

                a_random_mean = posterior_samples["a_random_mean"]
                a_random_scale = posterior_samples["a_random_scale"]
                np.save(os.path.join(model.build_dir, "a_random_mean.npy"), a_random_mean)
                np.save(os.path.join(model.build_dir, "a_random_scale.npy"), a_random_scale)

                config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
                model, posterior_samples, posterior_predictive = None, None, None
                a_true, a_pred = None, None
                a_random_mean, a_random_scale = None, None
                del config, df, prediction_df, encoder_dict, _
                del model, posterior_samples, posterior_predictive
                del a_true, a_pred
                del a_random_mean, a_random_scale
                gc.collect()

            # Non-hierarchical methods like Non Hierarchical Bayesian Model and
            # Maximum Likelihood Model need to be run separately on individual subjects
            # otherwise, there are convergence issues when the number of subjects is large
            case "non_hierarchical_bayesian_model" | "maximum_likelihood_model":
                for subject in range(n_subjects):
                    sub_dir = f"subject{subject}"

                    # Load data
                    ind = (
                        (simulation_df[simulator.features[0]] == subject) &
                        (simulation_df[REP] < n_reps)
                    )
                    df = simulation_df[ind].reset_index(drop=True).copy()
                    df[simulator.response[0]] = ppd_obs[draw, ind, 0]

                    ind = df[simulator.response[0]] > 0
                    df = df[ind].reset_index(drop=True).copy()

                    # Build model
                    config = Config(toml_path=TOML_PATH)
                    config.BUILD_DIR = os.path.join(
                        build_dir,
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
                    a_true = ppd_a[draw, [subject], ...]
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

            # This is also a non-hierarchical method. Internally, it will
            # run separately on individual subjects
            case "nelder_mead_optimization":
                # Load data
                ind = (
                    (simulation_df[simulator.features[0]] < n_subjects) &
                    (simulation_df[REP] < n_reps)
                )
                df = simulation_df[ind].reset_index(drop=True).copy()
                df[simulator.response[0]] = ppd_obs[draw, ind, 0]

                ind = df[simulator.response[0]] > 0
                df = df[ind].reset_index(drop=True).copy()

                # Build model
                config = Config(toml_path=TOML_PATH)
                config.BUILD_DIR = os.path.join(
                    build_dir,
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
                params = model.run_inference(df=df)

                # Predictions and recruitment curves
                prediction_df = model.make_prediction_dataset(df=df)
                prediction_df = model.predict(df=prediction_df, params=params)
                model.render_recruitment_curves(
                    df=df,
                    encoder_dict=encoder_dict,
                    params=params,
                    prediction_df=prediction_df,
                )

                # Compute error and save results
                a_true = ppd_a[draw, :n_subjects, ...]
                a_pred = params[site.a]
                assert a_pred.shape == a_true.shape
                np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
                np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

                config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
                model, params = None, None
                a_true, a_pred = None, None
                del config, df, prediction_df, encoder_dict, _
                del model, params
                del a_true, a_pred
                gc.collect()

            case _:
                raise ValueError(f"Invalid model {M.NAME}.")

        return


    # Experiment space
    draws_space = range(0, 1000)
    n_subjects_space = N_SUBJECTS_SPACE
    n_jobs = -1

    ## Uncomment the following to run
    ## experiment for different models

    # Run for Hierarchical Bayesian Model /
    models = [HierarchicalBayesianModel]

    # # Run for Non-hierarchical Bayesian Model
    # n_subjects_space = [24]
    # models = [NonHierarchicalBayesianModel]

    # # Run for Maximum Likelihood Model
    # n_subjects_space = [16]
    # models = [MaximumLikelihoodModel]

    # # Run for Nelder-Mead Optimization
    # n_subjects_space = [16]
    # models = [NelderMeadOptimization]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(
                N_REPS, N_PULSES, n_subjects, draw, M
            )
            for draw in draws_space
            for n_subjects in n_subjects_space
            for M in models
        )


if __name__ == "__main__":
    # # Run for simulation with effect
    # main(SIMULATE_DATA_DIR, EXPERIMENTS_DIR)

    # Run for simulation without effect
    main(SIMULATE_DATA_NO_EFFECT_DIR, EXPERIMENTS_NO_EFFECT_DIR)
