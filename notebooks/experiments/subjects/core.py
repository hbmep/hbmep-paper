import functools
import os
import gc
import pickle
import logging
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from joblib import Parallel, delayed

import arviz as az
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from simulate_data import Simulator

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()

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


class HBModel(BaseModel):
    NAME = "hbm"

    def __init__(self, config: Config):
        super(HBModel, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)
        n_fixed = 1
        n_random = n_features[1] - 1

        """ Fixed Effects (Baseline) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_fixed", n_fixed):
                a_fixed_mean = numpyro.sample("a_fixed_mean", dist.TruncatedNormal(50., 20., low=0))
                a_fixed_scale = numpyro.sample("a_fixed_scale", dist.HalfNormal(30.))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_fixed = numpyro.sample(
                        "a_fixed", dist.TruncatedNormal(a_fixed_mean, a_fixed_scale, low=0)
                    )

        """ Random Effects (Delta) """
        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("n_random", n_random):
                a_random_mean = numpyro.sample("a_random_mean", dist.Normal(0, 50))
                a_random_scale = numpyro.sample("a_random_scale", dist.HalfNormal(50.0))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    a_random = numpyro.sample("a_random", dist.Normal(a_random_mean, a_random_scale))

                    """ Penalty """
                    penalty_for_negative_a = (jnp.fabs(a_fixed + a_random) - (a_fixed + a_random))
                    numpyro.factor("penalty_for_negative_a", -penalty_for_negative_a)

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(5))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
            ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(10))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5))

            c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5))
            c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5))

            with numpyro.plate(site.n_features[1], n_features[1]):
                """ Hyper-priors """
                b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(ell_scale_global_scale))
                H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=-2)
                    )

                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                    v = numpyro.sample(site.v, dist.HalfNormal(v_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                    ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        v=v[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    c_1[feature0, feature1] + jnp.true_divide(c_2[feature0, feature1], mu)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=jnp.multiply(mu, beta), rate=beta),
                    obs=response_obs
                )


class NHBModel(BaseModel):
    NAME = "nhbm"

    def __init__(self, config: Config):
        super(NHBModel, self).__init__(config=config)

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        feature1 = features[1].reshape(-1,)

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Global Priors """
                    b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(5))
                    v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(5))

                    L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.5))
                    ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(10))
                    H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(5))

                    c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5))
                    c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5))

                    """ Hyper-priors """
                    a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(50., 20., low=0))
                    a_scale = numpyro.sample("a_scale", dist.HalfNormal(30.))

                    b_scale = numpyro.sample("b_scale", dist.HalfNormal(b_scale_global_scale))
                    v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

                    L_scale = numpyro.sample("L_scale", dist.HalfNormal(L_scale_global_scale))
                    ell_scale = numpyro.sample("ell_scale", dist.HalfNormal(ell_scale_global_scale))
                    H_scale = numpyro.sample("H_scale", dist.HalfNormal(H_scale_global_scale))

                    c_1_scale = numpyro.sample("c_1_scale", dist.HalfNormal(c_1_scale_global_scale))
                    c_2_scale = numpyro.sample("c_2_scale", dist.HalfNormal(c_2_scale_global_scale))

                    """ Priors """
                    a = numpyro.sample(
                        "a", dist.TruncatedNormal(a_mean, a_scale, low=0)
                    )

                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))
                    v = numpyro.sample(site.v, dist.HalfNormal(v_scale))

                    L = numpyro.sample(site.L, dist.HalfNormal(L_scale))
                    ell = numpyro.sample(site.ell, dist.HalfNormal(ell_scale))
                    H = numpyro.sample(site.H, dist.HalfNormal(H_scale))

                    c_1 = numpyro.sample(site.c_1, dist.HalfNormal(c_1_scale))
                    c_2 = numpyro.sample(site.c_2, dist.HalfNormal(c_2_scale))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        v=v[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    c_1[feature0, feature1] + jnp.true_divide(c_2[feature0, feature1], mu)
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=jnp.multiply(mu, beta), rate=beta),
                    obs=response_obs
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


    def _process(n_subjects, draw, seed, m):
        n_subjects_dir, draw_dir, seed_dir = f"n{n_subjects}", f"d{draw}", f"s{seed}"
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

        """ Build model """
        toml_path = "/home/vishu/repos/hbmep-paper/configs/experiments/subjects.toml"
        config = Config(toml_path=toml_path)
        config.BUILD_DIR = os.path.join(simulator.build_dir, "models", m.NAME, draw_dir, n_subjects_dir, seed_dir)
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
        # config.MCMC_PARAMS["num_warmup"] = 4000
        # config.MCMC_PARAMS["num_samples"] = 2000
        # config.MCMC_PARAMS["num_chains"] = 2
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

    n_subjects_space = [1, 4, 8, 16]
    models = [NHBModel]
    n_jobs = -1

    # draws_space = draws_space[:2]
    # n_subjects_space = n_subjects_space[:2]
    # seeds_for_generating_subjects = seeds_for_generating_subjects[:2]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(_process)(n_subjects, draw, seed, m) \
            for draw in draws_space \
            for n_subjects in n_subjects_space \
            for seed in seeds_for_generating_subjects \
            for m in models
        )


if __name__ == "__main__":
    main()
