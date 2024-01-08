import functools
import os
import gc
import pickle
import logging
import multiprocessing
from pathlib import Path

import pandas as pd
import numpy as np
import jax
from joblib import Parallel, delayed

import numpyro
from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from simulate_data import Simulator
from models import HBModel, NHBModel

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
dest = "/home/vishu/logs/pulses-experiment.log"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(dest, mode="w"),
        logging.StreamHandler()
    ],
    force=True
)


@timing
def main():
    dir ="/home/vishu/repos/hbmep-paper/reports/experiments/subjects/simulate-data/a_random_mean_-2.5_a_random_scale_1.5"
    src = os.path.join(dir, "simulation_ppd.pkl")
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)
    obs = simulation_ppd[site.obs]
    A_TRUE = simulation_ppd[site.a]
    logger.info(f"Simulation PPD obs: {obs.shape}")
    logger.info(f"obs min: {obs.min()}, obs max: {obs.max()}")
    logger.info(f"obs negative: {np.sum(obs < 0)}")
    logger.info(f"obs zero: {(np.sum(obs == 0) / functools.reduce(lambda x, y: x * y, obs.shape)) * 100}")

    src = os.path.join(dir, "simulation_df.csv")
    simulation_df = pd.read_csv(src)


    def _process(n_pulses, draw, seed, m):
        n_pulses_dir, draw_dir, seed_dir = f"p{n_pulses}", f"d{draw}", f"s{seed}"
        n_subjects = 8
        subjects = \
            jax.random.choice(
                key=jax.random.PRNGKey(seed),
                a=np.arange(0, A_TRUE.shape[1], 1),
                shape=(n_subjects,),
                replace=False
            ) \
            .tolist()

        ind = simulation_df[simulator.features[0]].isin(subjects)
        df = simulation_df[ind].reset_index(drop=True).copy()
        df[simulator.response[0]] = obs[draw, ...][ind, 0]
        ind = df[simulator.response[0]] > 0
        df = df[ind].reset_index(drop=True).copy()

        ind = df[simulator.intensity].isin(
            UNQ_PULSES[map_n_pulses_to_pulses_index[n_pulses]]
        )
        df = df[ind].reset_index(drop=True).copy()

        """ Build model """
        toml_path = "/home/vishu/repos/hbmep-paper/configs/experiments/subjects.toml"
        config = Config(toml_path=toml_path)
        config.BUILD_DIR = os.path.join(simulator.build_dir, "pulses", m.NAME, draw_dir, n_pulses_dir, seed_dir)
        logger = logging.getLogger(__name__)
        FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        config.MCMC_PARAMS["num_warmup"] = 4000
        config.MCMC_PARAMS["num_samples"] = 2000
        config.MCMC_PARAMS["num_chains"] = 2
        model = m(config=config)

        df, encoder_dict = model.load(df=df)
        _, posterior_samples = model.run_inference(df=df)

        prediction_df = model.make_prediction_dataset(df=df)
        posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
        model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

        a_true = A_TRUE[draw, ...][sorted(subjects), ...]
        a_pred = posterior_samples[site.a]
        assert a_pred.mean(axis=0).shape == a_true.shape
        logger.info(f"A_TRUE: {A_TRUE.shape}, {type(A_TRUE)}")
        logger.info(f"a_true: {a_true.shape}, {type(a_true)}")
        logger.info(f"a_pred: {a_pred.shape}, {type(a_pred)}")
        np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
        np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

        if model.NAME == "hbm":
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
        return


    TOTAL_SUBJECTS = 200
    n_draws = 50
    n_repeats = 50
    keys = jax.random.split(simulator.rng_key, num=2)
    logger.info(f"Possible draws: {simulation_ppd[site.a].shape[0]}")
    draws_space = \
        jax.random.choice(
            key=keys[0],
            a=np.arange(0, simulation_ppd[site.a].shape[0], 1),
            shape=(n_draws,),
            replace=False
        ) \
        .tolist()
    logger.info(f"Possible seeds: {n_repeats * 100}")
    seeds_for_generating_subjects = \
        jax.random.choice(
            key=keys[1],
            a=np.arange(0, n_repeats * 100, 1),
            shape=(n_repeats,),
            replace=False
        ) \
        .tolist()

    n_pulses_space = [10, 20, 30, 40, 50]
    models = [HBModel]
    n_jobs = -1

    """ Fix the pulses for every n_pulses """
    UNQ_PULSES = simulation_df[simulator.intensity].unique()
    UNQ_PULSES = np.sort(UNQ_PULSES)
    TOTAL_PULSES = UNQ_PULSES.shape[0]
    assert TOTAL_PULSES == 60

    n_pulses_space_to_run = [10, 20, 30, 40, 50, 60]
    map_n_pulses_to_pulses_index = {
        TOTAL_PULSES: np.linspace(0, TOTAL_PULSES - 1, TOTAL_PULSES).astype(int).tolist()
    }
    logger.info(f"map_n_pulses_to_pulses_index: {map_n_pulses_to_pulses_index}")

    for i in range(len(n_pulses_space_to_run) - 1, -1, -1):
        n_pulses = n_pulses_space_to_run[i]
        logger.info(f"n_pulses: {n_pulses}")
        if n_pulses == TOTAL_PULSES: continue
        pulses_index_to_subsample_from = \
            map_n_pulses_to_pulses_index[n_pulses_space_to_run[i + 1]]
        ind = \
            np.round(np.linspace(0, len(pulses_index_to_subsample_from) - 1, n_pulses)) \
            .astype(int)
        map_n_pulses_to_pulses_index[n_pulses] = np.array(pulses_index_to_subsample_from)[ind]

    for i in range(len(n_pulses_space_to_run) - 1, -1, -1):
        n_pulses = n_pulses_space_to_run[i]
        logger.info(f"n_pulses: {n_pulses}, {map_n_pulses_to_pulses_index[n_pulses]}")
        if n_pulses != TOTAL_PULSES:
            assert set(map_n_pulses_to_pulses_index[n_pulses]) <= set(map_n_pulses_to_pulses_index[n_pulses_space_to_run[i + 1]])

    draws_space = draws_space[:2]
    n_pulses_space = n_pulses_space[:2]
    seeds_for_generating_subjects = seeds_for_generating_subjects[:2]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(_process)(n_pulses, draw, seed, m) \
            for draw in draws_space \
            for n_pulses in n_pulses_space \
            for seed in seeds_for_generating_subjects \
            for m in models
        )


if __name__ == "__main__":
    main()
