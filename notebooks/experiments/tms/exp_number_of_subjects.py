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

from models import Simulator, HBModel, NHBModel

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
dest = "/home/vishu/logs/subjects-experiment.log"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(dest, mode="w"),
        logging.StreamHandler()
    ],
    force=True
)

EXPERIMENT_NAME = "exp_number_of_subjects"
N_DRAWS = 50
N_REPEATS = 50


def fix_rng(rng_key, max_draws, max_seeds):
    keys = jax.random.split(rng_key, num=2)
    draws_space = \
        jax.random.choice(
            key=keys[0],
            a=np.arange(0, max_draws, 1),
            shape=(N_DRAWS,),
            replace=False
        ) \
        .tolist()
    logger.info(f"draws: {draws_space}")
    seeds_for_generating_subjects = \
        jax.random.choice(
            key=keys[1],
            a=np.arange(0, max_seeds, 1),
            shape=(N_REPEATS,),
            replace=False
        ) \
        .tolist()
    logger.info(f"seeds: {seeds_for_generating_subjects}")
    return draws_space, seeds_for_generating_subjects


@timing
def main():
    """ Load simulated data """
    dir ="/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate-data/a_random_mean_-2.5_a_random_scale_1.5"
    src = os.path.join(dir, "simulation_ppd.pkl")
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    ppd_obs = simulation_ppd[site.obs]
    ppd_a = simulation_ppd[site.a]

    src = os.path.join(dir, "simulation_df.csv")
    simulation_df = pd.read_csv(src)

    src = os.path.join(dir, "filter.npy")
    filter = np.load(src)

    """ Fix rng """
    rng_key = simulator.rng_key
    max_draws = ppd_a.shape[0]
    max_seeds = N_REPEATS * 100
    draws_space, seeds_for_generating_subjects = fix_rng(
        rng_key, max_draws, max_seeds
    )
    n_subjects_space = [1, 4, 8, 16]
    n_jobs = -1


    """ Define experiment """
    def run_experiment(
        n_subjects,
        draw,
        seed,
        M
    ):
        """ Artefacts directory """
        n_subjects_dir, draw_dir, seed_dir = \
            f"n{n_subjects}", f"d{draw}", f"s{seed}"

        """ Load data """
        valid_subjects = \
            np.arange(0, ppd_a.shape[1], 1)[filter[draw, ...]]
        subjects = \
            jax.random.choice(
                key=jax.random.PRNGKey(seed),
                a=valid_subjects,
                shape=(n_subjects,),
                replace=False
            ) \
            .tolist()

        if M.NAME in ["hbm"]:
            ind = simulation_df[simulator.features[0]].isin(subjects)
            df = simulation_df[ind].reset_index(drop=True).copy()
            df[simulator.response[0]] = ppd_obs[draw, ind, 0]
            ind = df[simulator.response[0]] > 0
            df = df[ind].reset_index(drop=True).copy()

            """ Build model """
            toml_path = "/home/vishu/repos/hbmep-paper/configs/experiments/tms.toml"
            config = Config(toml_path=toml_path)
            config.BUILD_DIR = os.path.join(simulator.build_dir, EXPERIMENT_NAME, draw_dir, n_subjects_dir, seed_dir, M.NAME)

            # Set up logging
            logger = logging.getLogger(__name__)
            dest = os.path.join(config.BUILD_DIR, "log.log")
            simulator._make_dir(config.BUILD_DIR)
            logging.basicConfig(
                format=FORMAT,
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(dest, mode="w"),
                    logging.StreamHandler()
                ],
                force=True
            )

            model = M(config=config)

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
                sub_dir = f"p{subject}"
                ind = simulation_df[simulator.features[0]].isin([subject])
                df = simulation_df[ind].reset_index(drop=True).copy()
                df[simulator.response[0]] = ppd_obs[draw, ind, 0]
                ind = df[simulator.response[0]] > 0
                df = df[ind].reset_index(drop=True).copy()

                """ Build model """
                toml_path = "/home/vishu/repos/hbmep-paper/configs/experiments/tms.toml"
                config = Config(toml_path=toml_path)
                config.BUILD_DIR = os.path.join(simulator.build_dir, EXPERIMENT_NAME, draw_dir, n_subjects_dir, seed_dir, M.NAME, sub_dir)

                # Set up logging
                logger = logging.getLogger(__name__)
                dest = os.path.join(config.BUILD_DIR, "log.log")
                simulator._make_dir(config.BUILD_DIR)
                logging.basicConfig(
                    format=FORMAT,
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(dest, mode="w"),
                        logging.StreamHandler()
                    ],
                    force=True
                )

                model = M(config=config)

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


    draws_space = draws_space[:30]
    # seeds_for_generating_subjects = seeds_for_generating_subjects[:10]

    # Run for Hierarchical Bayesian Model
    models = [HBModel]

    # # Run for Non-Hierarchical Bayesian Model
    # n_subjects_space = [16]
    # models = [NHBModel]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(
                n_subjects, draw, seed, M
            ) \
            for draw in draws_space \
            for n_subjects in n_subjects_space \
            for seed in seeds_for_generating_subjects \
            for M in models
        )


if __name__ == "__main__":
    logger.info(f"Logging to {dest}")
    main()
