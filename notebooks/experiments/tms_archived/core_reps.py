import os
import gc
import pickle
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jax
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from models import Simulator, HBModel, NHBModel
from hbmep_paper.utils import setup_logging
from utils import fix_draws_and_seeds, fix_nested_pulses
from constants import TOML_PATH, N_DRAWS, N_SEEDS

logger = logging.getLogger(__name__)

SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_-3.0_a_random_scale_1.5"
SIMULATION_DF_PATH = os.path.join(SIMULATION_DIR, "simulation_df.csv")
SIMULATION_PARAMS_PATH = os.path.join(SIMULATION_DIR, "simulation_params.pkl")
MASK_PATH = os.path.join(SIMULATION_DIR, "mask.npy")

EXPERIMENT_NAME = "number_of_reps"
N_SUBJECTS = 8
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate/a_random_mean_-3.0_a_random_scale_1.5/experiments/"


@timing
def main():
    """ Load simulated dataframe """
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    """ Load simulation params """
    src = SIMULATION_PARAMS_PATH
    with open(src, "rb") as g:
        simulator, simulation_params = pickle.load(g)

    simulator._make_dir(BUILD_DIR)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )

    # """ Load simulation ppd """
    # src = SIMULATION_PPD_PATH
    # with open(src, "rb") as g:
    #     simulation_ppd, = pickle.load(g)

    """ Load mask """
    src = MASK_PATH
    mask = np.load(src)

    """ Fix draws and seeds """
    rng_key = simulator.rng_key
    max_draws = simulation_params[site.a].shape[0]
    max_seeds = N_SEEDS * 100
    draws_space, seeds_for_generating_subjects = fix_draws_and_seeds(
        rng_key, max_draws, max_seeds
    )

    """ Fix pulses """
    pulses_map = fix_nested_pulses(simulator, simulation_df)

    """ Experiment space """
    n_subjects = N_SUBJECTS

    n_pulses_space = [24, 32, 40, 48]
    n_reps_space = [1, 2, 4, 8]
    n_jobs = -1

    ppd_a = simulation_params[site.a]

    # """ Visualize nested pulses set """
    # nrows, ncols = 1, 1
    # fig, axes = plt.subplots(
    #     nrows=nrows,
    #     ncols=ncols,
    #     figsize=(ncols * 5, nrows * 3),
    #     squeeze=False,
    #     constrained_layout=True
    # )
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(list(pulses_map.keys()))))
    # for i, n_pulses in enumerate(list(pulses_map.keys())):
    #     pulses = pulses_map[n_pulses]
    #     ax = axes[0, 0]
    #     sns.scatterplot(
    #         x=[n_pulses] * n_pulses,
    #         y=pulses,
    #         color=colors[i],
    #         ax=ax
    #     )
    # dest = os.path.join(simulator.build_dir, "nested_pulses.png")
    # fig.savefig(dest)
    # logger.info(f"Saved to {dest}")


    """ Define experiment """
    def run_experiment(
        n_reps,
        n_pulses,
        n_subjects,
        draw,
        seed,
        M
    ):
        """ Artefacts directory """
        n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
        draw_dir, seed_dir = f"d{draw}", f"s{seed}"

        """ Load data """
        valid_subjects = \
            np.arange(0, ppd_a.shape[1], 1)[mask[draw, ...]]
        subjects = \
            jax.random.choice(
                key=jax.random.PRNGKey(seed),
                a=valid_subjects,
                shape=(n_subjects,),
                replace=False
            ) \
            .tolist()

        pulses = pulses_map[n_pulses]
        pulses = np.array(pulses)[::n_reps]
        pulses = list(pulses)

        if M.NAME in ["hbm"]:
            ind = simulation_df[simulator.features[0]].isin(subjects)
            df = simulation_df[ind].reset_index(drop=True).copy()
            df = \
                pd.concat([df] * n_reps, ignore_index=True) \
                .reset_index(drop=True) \
                .copy()

            ppd_obs = []
            for r in range(n_reps):
                src = os.path.join(SIMULATION_DIR, f"simulation_ppd_{r}.pkl")
                with open(src, "rb") as g:
                    curr_obs, = pickle.load(g)
                curr_obs = curr_obs[site.obs]
                curr_obs = curr_obs[draw, ind, 0]
                ppd_obs += list(curr_obs)

            df[simulator.response[0]] = np.array(ppd_obs)
            curr_obs = None
            ppd_obs = None

            ind = df[simulator.response[0]] > 0
            df = df[ind].reset_index(drop=True).copy()

            """ Filter pulses"""
            ind = df[simulator.intensity].isin(pulses)
            df = df[ind].reset_index(drop=True).copy()
            # assert df[simulator.intensity].unique().shape[0] == n_pulses

            """ Build model """
            toml_path = TOML_PATH
            config = Config(toml_path=toml_path)
            config.BUILD_DIR = os.path.join(
                BUILD_DIR,
                EXPERIMENT_NAME,
                draw_dir,
                n_subjects_dir,
                n_reps_dir,
                n_pulses_dir,
                seed_dir,
                M.NAME
            )
            model = M(config=config)

            # Set up logging
            model._make_dir(model.build_dir)
            setup_logging(
                dir=model.build_dir,
                fname="logs"
            )

            """ Run inference """
            df, encoder_dict = model.load(df=df)
            _, posterior_samples = model.run_inference(df=df)

            """ Predictions and recruitment curves """
            prediction_df = model.make_prediction_dataset(df=df)
            posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
            model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

            """ Compute error and save results """
            a_true = ppd_a[draw, ...][sorted(subjects), ...]
            a_pred = posterior_samples[site.a]
            assert a_pred.mean(axis=0).shape == a_true.shape
            np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
            np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

            a_random_mean = posterior_samples["a_random_mean"]
            a_random_scale = posterior_samples["a_random_scale"]
            logger.info(f"a_random_mean: {a_random_mean.shape}, {type(a_random_mean)}")
            logger.info(f"a_random_scale: {a_random_scale.shape}, {type(a_random_scale)}")
            np.save(os.path.join(model.build_dir, "a_random_mean.npy"), a_random_mean)
            np.save(os.path.join(model.build_dir, "a_random_scale.npy"), a_random_scale)

            config, df, prediction_df, encoder_dict, _,  = None, None, None, None, None
            model, posterior_samples, posterior_predictive = None, None, None
            results, a_true, a_pred, a_random_mean, a_random_scale = None, None, None, None, None
            del config, df, prediction_df, encoder_dict, _
            del model, posterior_samples, posterior_predictive
            del results, a_true, a_pred, a_random_mean, a_random_scale
            gc.collect()

        # Non-hierarchical Bayesian model needs to be run separately on individual subjects
        # otherwise, there are convergence issues when the number of subjects is large
        elif M.NAME in ["nhbm"]:
            for subject in subjects:
                sub_dir = f"subject{subject}"
                ind = simulation_df[simulator.features[0]].isin([subject])
                df = simulation_df[ind].reset_index(drop=True).copy()
                df = \
                    pd.concat([df] * n_reps, ignore_index=True) \
                    .reset_index(drop=True) \
                    .copy()

                ppd_obs = []
                for r in range(n_reps):
                    src = os.path.join(SIMULATION_DIR, f"simulation_ppd_{r}.pkl")
                    with open(src, "rb") as g:
                        curr_obs, = pickle.load(g)
                    curr_obs = curr_obs[site.obs]
                    curr_obs = curr_obs[draw, ind, 0]
                    ppd_obs += list(curr_obs)

                df[simulator.response[0]] = np.array(ppd_obs)
                curr_obs = None
                ppd_obs = None

                ind = df[simulator.response[0]] > 0
                num_num_positive = ind.shape[0] - ind.sum()
                df = df[ind].reset_index(drop=True).copy()

                """ Filter pulses"""
                ind = df[simulator.intensity].isin(pulses)
                df = df[ind].reset_index(drop=True).copy()
                # assert np.abs(df.shape[0] - 2 * n_pulses) < 6

                """ Build model """
                toml_path = TOML_PATH
                config = Config(toml_path=toml_path)
                config.BUILD_DIR = os.path.join(
                    BUILD_DIR,
                    EXPERIMENT_NAME,
                    draw_dir,
                    n_subjects_dir,
                    n_reps_dir,
                    n_pulses_dir,
                    seed_dir,
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

                """ Run inference """
                df, encoder_dict = model.load(df=df)
                _, posterior_samples = model.run_inference(df=df)

                """ Predictions and recruitment curves """
                prediction_df = model.make_prediction_dataset(df=df)
                posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
                model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

                """ Compute error and save results """
                a_true = ppd_a[draw, ...][[subject], ...]
                a_pred = posterior_samples[site.a]
                assert a_pred.mean(axis=0).shape == a_true.shape
                np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
                np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

                config, df, prediction_df, encoder_dict, _,  = None, None, None, None, None
                model, posterior_samples, posterior_predictive = None, None, None
                results, a_true, a_pred, a_random_mean, a_random_scale = None, None, None, None, None
                del config, df, prediction_df, encoder_dict, _
                del model, posterior_samples, posterior_predictive
                del results, a_true, a_pred, a_random_mean, a_random_scale
                gc.collect()

        return


    models = [HBModel, NHBModel]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(
                n_reps, n_pulses, n_subjects, draw, seed, M
            ) \
            for draw in draws_space \
            for n_reps in n_reps_space \
            for n_pulses in n_pulses_space \
            for seed in seeds_for_generating_subjects \
            for M in models
        )


if __name__ == "__main__":

    main()