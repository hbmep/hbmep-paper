import os
import gc
import time
import shutil
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
from models import ActiveReLU, ReLU
from utils import generate_nested_pulses
from constants import (TOML_PATH, REP)

logger = logging.getLogger(__name__)

SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-active-learn/simulate_data"
SIMULATION_DF_PATH = os.path.join(SIMULATION_DIR, "simulation_df.csv")
SIMULATION_PPD_PATH = os.path.join(SIMULATION_DIR, "simulation_ppd.pkl")
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-active-learn/simulate_data/relu-active"

N_SUBJECTS, N_REPS, N_OBS = 1, 1, 20


def main():
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    ind = \
        (simulation_df[simulator.features[0]] < N_SUBJECTS) & \
        (simulation_df[REP] < N_REPS)

    simulation_df = simulation_df[ind].reset_index(drop=True).copy()
    ppd_a = simulation_ppd[site.a][:, 0, :]
    ppd_obs = simulation_ppd[site.obs][:, ind, :]
    # simulation_df[simulator.response[0]] = ppd_obs[0, ...]

    simulation_ppd = None
    del simulation_ppd
    gc.collect()

    # Generate nested pulses
    pulses_map = generate_nested_pulses(simulator, simulation_df)

    # Set up logging
    os.makedirs(BUILD_DIR, exist_ok=True)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    logger.info(f"Simulation dataframe shape: {simulation_df.shape}")
    logger.info(f"Simulation ppd shape: {ppd_a.shape}, {ppd_obs.shape}")


    # Define experiment
    def run_experiment(
        method,
        draw,
        M,
        n_reps,
        n_pulses,
        n_subjects

    ):
        # Artefacts directory
        n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
        draw_dir = f"d{draw}"

        ind = simulation_df[simulator.intensity].isin(pulses_map[n_pulses])
        df = simulation_df[ind].reset_index(drop=True).copy()
        df[simulator.response[0]] = ppd_obs[draw, ind, 0]

        ind = df[simulator.response[0]] > 0
        df = df[ind].reset_index(drop=True).copy()

        # Build model
        toml_path = TOML_PATH
        config = Config(toml_path=toml_path)
        config.BUILD_DIR = os.path.join(
            BUILD_DIR,
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

        df, _ = model.load(df=df)
        intensity, = model._collect_regressor(df=df)
        response_obs, = model._collect_response(df=df)

        # Run inference
        match method:
            case "mcmc":
                posterior_samples, time_taken_pre = model.run_inference(intensity, response_obs)
            case "svi":
                posterior_samples, time_taken_pre = model.run_svi(intensity, response_obs)
            case _:
                raise ValueError(f"Invalid method: {method}")

        prediction_df = model.make_prediction_dataset(df=df, num=model.n_grid)
        posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
        intensity_pred, = model._collect_regressor(df=prediction_df)
        dest = os.path.join(model.build_dir, "pre.png")
        model.render_recruitment_curves(
            intensity=intensity,
            response=response_obs,
            intensity_pred=intensity_pred,
            a=posterior_samples[site.a],
            mu=posterior_predictive[site.mu],
            obs=posterior_predictive[site.obs],
            destination_path=dest
        )

        # Compute error and save results
        a_true = ppd_a[draw, :n_subjects, ...]
        a_pred = posterior_samples[site.a][..., 0]
        assert a_pred.mean(axis=0).shape == a_true.shape
        np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
        np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)
        logger.info(f"Pre: Time taken: {time_taken_pre} seconds")
        logger.info(f"Pre: MAE: {np.abs(a_true.mean() - a_pred.mean(axis=0)).mean()}")

        logger.info(f"Shuffling and keeping only {N_OBS} draws ...")
        ind = np.arange(0, posterior_predictive[site.obs].shape[0], 1)
        ind = jax.random.permutation(model.rng_key, ind)
        ind = np.array(ind)
        posterior_samples = {
            u: v[ind, ...][:N_OBS, ...] for u, v in posterior_samples.items()
        }
        posterior_predictive = {
            u: v[ind, ...][:N_OBS, ...] for u, v in posterior_predictive.items()
        }
        # for u, v in posterior_samples.items():
        #     logger.info(f"{u}: {v.shape}")
        # for u, v in posterior_predictive.items():
        #     logger.info(f"{u}: {v.shape}")

        obs = posterior_predictive[site.obs]
        intensity_new = intensity_pred.reshape(-1,)
        intensity_new = np.append(
            np.tile(intensity, (1, 1, model.n_grid)),
            intensity_new[None, None, :],
            axis=0
        )
        intensity_new = intensity_new[None, ...]
        intensity_new = np.tile(intensity_new, (posterior_predictive[site.obs].shape[0], 1, 1, 1))
        response_obs_new = response_obs[None, ...]
        response_obs_new = np.tile(response_obs_new, (obs.shape[0], 1, 1, model.n_grid))
        response_obs_new = np.append(
            response_obs_new,
            np.swapaxes(obs, 1, -1),
            axis=1
        )
        intensity_new = np.moveaxis(intensity_new, 0, -1)
        response_obs_new = np.moveaxis(response_obs_new, 0, -1)
        intensity_new = intensity_new.reshape(*intensity_new.shape[:-2], -1)
        response_obs_new = response_obs_new.reshape(*response_obs_new.shape[:-2], -1)
        # intensity_new = intensity_new[..., :5]
        # response_obs_new = response_obs_new[..., :5]
        logger.info(f"intensity_new: {intensity_new.shape}")
        logger.info(f"response_obs_new: {response_obs_new.shape}")

        # Run inference
        match method:
            case "mcmc":
                def _run_inference(regression_ind):
                    regression_dir = os.path.join(model.build_dir, f"regression_{regression_ind}")
                    os.makedirs(regression_dir, exist_ok=True)
                    posterior_samples, _ = model.run_inference(intensity_new[..., regression_ind], response_obs_new[..., regression_ind])
                    dest = os.path.join(regression_dir, "inference.pkl")
                    with open(dest, "wb") as g:
                        pickle.dump((posterior_samples,), g)

                start = time.time()
                with Parallel(n_jobs=-1) as parallel:
                    parallel(
                        delayed(_run_inference)(regression_ind)
                        for regression_ind in range(intensity_new.shape[-1])
                    )
                end = time.time()
                time_taken_post = end - start

                posterior_samples = None
                for regression_ind in range(intensity_new.shape[-1]):
                    regression_dir = os.path.join(model.build_dir, f"regression_{regression_ind}")
                    src = os.path.join(regression_dir, "inference.pkl")
                    with open(src, "rb") as g:
                        _posterior_samples, = pickle.load(g)
                    if posterior_samples is None:
                        posterior_samples = _posterior_samples
                    else:
                        for u, v in _posterior_samples.items():
                            posterior_samples[u] = np.concatenate([posterior_samples[u], v], axis=-1)

                for regression_ind in range(intensity_new.shape[-1]):
                    regression_dir = os.path.join(model.build_dir, f"regression_{regression_ind}")
                    shutil.rmtree(regression_dir, ignore_errors=True)
                # for u, v in posterior_samples.items():
                #     logger.info(f"{u}: {v.shape}")

                regression_dir, _posterior_samples = None, None
                del regression_dir, _posterior_samples
                gc.collect()

            case "svi":
                posterior_samples, time_taken_post = model.run_svi(intensity_new, response_obs_new)

            case _:
                raise ValueError(f"Invalid method: {method}")
        logger.info(f"Post: Time taken: {time_taken_post} seconds")
        np.save(os.path.join(model.build_dir, "time_taken.npy"), np.array([time_taken_pre, time_taken_post]))

        posterior_samples = {
            u: v[..., ::N_OBS][..., ::10]
            for u, v in posterior_samples.items()
        }
        posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
        # for u, v in posterior_predictive.items():
        #     logger.info(f"{u}: {v.shape}")
        dest = os.path.join(model.build_dir, "post.png")
        model.render_recruitment_curves(
            intensity=intensity_new[..., ::N_OBS][..., ::10],
            response=response_obs_new[..., ::N_OBS][..., ::10],
            intensity_pred=np.tile(intensity_pred, (1, 1, 10)),
            a=posterior_samples[site.a],
            mu=posterior_predictive[site.mu],
            obs=posterior_predictive[site.obs],
            destination_path=dest,
            if_color_last=True
        )


        ind, df, config, model, intensity, response_obs, posterior_samples, time_taken_pre, prediction_df, posterior_predictive, intensity_pred, a_true, a_pred, time_taken_post, obs, intensity_new, response_obs_new, = tuple([None] * 17)
        del ind, df, config, model, intensity, response_obs, posterior_samples, time_taken_pre, prediction_df, posterior_predictive, intensity_pred, a_true, a_pred, time_taken_post, obs, intensity_new, response_obs_new
        gc.collect()


    # # Run experiment
    # run_experiment(
    #     method="mcmc",
    #     draw=0,
    #     M=ActiveReLU,
    #     n_reps=N_REPS,
    #     n_pulses=48,
    #     n_subjects=N_SUBJECTS
    # )

    n_draws_space = range(ppd_obs.shape[0])
    n_draws_space = range(50)
    for draw in n_draws_space:
        run_experiment(
            method="mcmc",
            draw=draw,
            M=ActiveReLU,
            n_reps=N_REPS,
            n_pulses=48,
            n_subjects=N_SUBJECTS
        )


if __name__ == "__main__":
    main()
