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
from models import (
    Simulator,
    HierarchicalBayesianModel,
    # NonHierarchicalBayesianModel,
    # MaximumLikelihoodModel,
    # NelderMeadOptimization
)
from utils import _generate_simulation_data_dirs
from constants import (
    TOML_PATH,
    N_SUBJECTS_SPACE,
    TOTAL_REPS,
    TOTAL_PULSES,
    REP,
    INFERENCE_FILE,
    SIMULATION_DF,
)

logger = logging.getLogger(__name__)

N_REPS = TOTAL_REPS
N_PULSES = TOTAL_PULSES


@timing
def main(
    simulation_data_dir,
	draws_space,
	n_subjects_space,
	models,
	n_jobs=-1
):
    build_dir = os.path.join(simulation_data_dir, "experiments")

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
                df[simulator.response] = ppd_obs[draw, ind, :]

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

                a_delta_loc = posterior_samples["a_delta_loc"]
                a_delta_scale = posterior_samples["a_delta_scale"]
                a_delta_loc_loc = posterior_samples["a_delta_loc_loc"]
                np.save(os.path.join(model.build_dir, "a_delta_loc.npy"), a_delta_loc)
                np.save(os.path.join(model.build_dir, "a_delta_scale.npy"), a_delta_scale)
                np.save(os.path.join(model.build_dir, "a_delta_loc_loc.npy"), a_delta_loc_loc)

                config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
                model, posterior_samples, posterior_predictive = None, None, None
                a_true, a_pred = None, None
                a_delta_loc, a_delta_scale, a_delta_loc_loc = None, None, None
                del config, df, prediction_df, encoder_dict, _
                del model, posterior_samples, posterior_predictive
                del a_true, a_pred
                del a_delta_loc, a_delta_scale, a_delta_loc_loc
                gc.collect()

            # Non-hierarchical models: non-hierarchical Bayesian
            # and Maximum Likelihood
            case "non_hierarchical_bayesian_model" | "maximum_likelihood_model":
                for subject in range(n_subjects):
                    for intervention in range(2):
                        sub_dir = f"subject{subject}"
                        intervention_dir = f"inter{intervention}"

                        # Load data
                        ind = (
                            (simulation_df[simulator.features[0]] == subject) &
                            (simulation_df[simulator.features[1]] == intervention) &
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
                            sub_dir,
                            intervention_dir
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
                        # a_true = ppd_a[draw, [subject], ...]
                        a_true = ppd_a[draw, [subject], ...]
                        a_true = a_true[:, [intervention], ...]
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
            # run separately on individual recruitment curves
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


    logger.info("Power analysis experiment.")
    logger.info(f"build_dir: {build_dir}")
    logger.info(f"n_subjects_space: {', '.join(map(str, n_subjects_space))}")
    logger.info(f"models: {', '.join([z.NAME for z in models])}")
    logger.info(f"Running draws {draws_space.start} to {draws_space.stop - 1}.")
    logger.info(f"n_jobs: {n_jobs}")

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
    # Usage: python -m core_power 0 2000
    lo, hi = list(map(int, sys.argv[1:]))
    simulation_data_dirs = _generate_simulation_data_dirs()

    # Experiment space
    draws_space = range(lo, hi)
    n_jobs = -1

    # Uncomment the following to run
    # experiment for different models

    # Run hierarchical models
    n_subjects_space = N_SUBJECTS_SPACE
    models = [
        HierarchicalBayesianModel
    ]

    # # Run non-hierarchical models including
    # # non-hierarchical Bayesian and Maximum Likelihood
    # n_subjects_space = N_SUBJECTS_SPACE[-1:]
    # models = [
    #     NonHierarchicalBayesianModel,
    #     MaximumLikelihoodModel
    # ]

    # # Run non-hierarchical Nelder-Mead optimization
    # n_subjects_space = N_SUBJECTS_SPACE[-1:]
    # models = [NelderMeadOptimization]

    key = "with_no_effect"
    _, _, simulation_data_dir = simulation_data_dirs[key]
    # Run for simulation without effect
    main(
        simulation_data_dir=simulation_data_dir,
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        models=models,
        n_jobs=n_jobs
    )

    # main(
    #     simulation_data_dir=SIMULATE_DATA_WITH_NO_EFFECT_DIR,
    #     build_dir=EXPERIMENTS_WITH_NO_EFFECT_DIR,
    #     draws_space=draws_space,
    #     n_subjects_space=n_subjects_space,
    #     models=models,
    #     n_jobs=n_jobs
    # )
