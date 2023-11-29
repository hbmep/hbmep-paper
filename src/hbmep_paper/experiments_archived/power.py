import os
import gc
import pickle
import logging
import multiprocessing
from pathlib import Path

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from joblib import Parallel, delayed

import numpyro
import numpyro.distributions as dist
from hbmep.model import Baseline
from hbmep_paper.utils.constants import HBM

from hbmep.config import Config
from hbmep.model.utils import Site as site

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# formatter = logging.Formatter(FORMAT)

# fhandler = logging.FileHandler(filename='power-analysis-parallel.log', mode="w")
# fhandler.setFormatter(formatter)
# fhandler.setLevel(logging.INFO)
# logger.addHandler(fhandler)


class HierarchicalBayesianModel(Baseline):
    LINK = HBM

    def __init__(self, config: Config):
        super(HierarchicalBayesianModel, self).__init__(config=config)
        self.combination_columns = self.features + [self.subject]
        self.priors = {"baseline", "delta", site.a, site.b, site.L, site.H, site.v, site.g_1, site.g_2}

    def _model(self, subject, features, intensity, response_obs=None):
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_data = intensity.shape[0]
        n_subject = np.unique(subject).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        n_baseline = 1
        n_delta = n_feature0 - 1

        """ Baseline """
        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_baseline", n_baseline, dim=-2):
                mu_baseline = numpyro.sample(
                    "mu_baseline",
                    dist.TruncatedNormal(5, 10, low=0)
                )
                sigma_baseline = numpyro.sample(
                    "sigma_baseline",
                    dist.HalfNormal(10.0)
                )

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    baseline = numpyro.sample(
                        "baseline",
                        dist.TruncatedNormal(mu_baseline, sigma_baseline, low=0)
                    )

        """ Delta """
        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_delta", n_delta, dim=-2):
                mu_delta = numpyro.sample("mu_delta", dist.Normal(0, 10))
                sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(10.0))

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    delta = numpyro.sample("delta", dist.Normal(mu_delta, sigma_delta))

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_feature0", n_feature0, dim=-2):
                """ Hyper-priors """
                sigma_b = numpyro.sample(site.sigma_b, dist.HalfNormal(10))

                sigma_L = numpyro.sample(site.sigma_L, dist.HalfNormal(2))
                sigma_H = numpyro.sample(site.sigma_H, dist.HalfNormal(10))
                sigma_v = numpyro.sample(site.sigma_v, dist.HalfNormal(10))

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    """ Priors """
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([baseline, baseline + delta], axis=1)
                    )
                    b = numpyro.sample(site.b, dist.HalfNormal(sigma_b))

                    L = numpyro.sample(site.L, dist.HalfNormal(sigma_L))
                    H = numpyro.sample(site.H, dist.HalfNormal(sigma_H))
                    v = numpyro.sample(site.v, dist.HalfNormal(sigma_v))

                    g_1 = numpyro.sample(site.g_1, dist.Exponential(0.01))
                    g_2 = numpyro.sample(site.g_2, dist.Exponential(0.01))

        """ Model """
        mu = numpyro.deterministic(
            site.mu,
            L[subject, feature0]
            + jnp.maximum(
                0,
                -1
                + (H[subject, feature0] + 1)
                / jnp.power(
                    1
                    + (jnp.power(1 + H[subject, feature0], v[subject, feature0]) - 1)
                    * jnp.exp(-b[subject, feature0] * (intensity - a[subject, feature0])),
                    1 / v[subject, feature0]
                )
            )
        )
        beta = numpyro.deterministic(
            site.beta,
            g_1[subject, feature0] + g_2[subject, feature0] * (1 / mu) ** 2
        )

        """ Penalty """
        penalty = (jnp.fabs(baseline + delta) - (baseline + delta))
        numpyro.factor("penalty", -penalty)

        """ Observation """
        with numpyro.plate(site.data, n_data):
            return numpyro.sample(
                site.obs,
                dist.Gamma(concentration=mu * beta, rate=beta).to_event(1),
                obs=response_obs
            )


if __name__ == "__main__":
    def _process(N_counter, draw_counter, repeat_counter):
        N = N_space[N_counter]
        draw_ind = draws_space[draw_counter]
        seed = repeats_space[repeat_counter]

        N_dir, draw_dir, seed_dir = f"N_{N}", f"draw_{draw_ind}", f"seed_{seed}"

        subjects_ind = \
            jax.random.choice(
                key=jax.random.PRNGKey(seed),
                a=np.arange(0, TOTAL_SUBJECTS, 1),
                shape=(N,),
                replace=False
            ) \
            .tolist()

        ind = PREDICTION_DF[MODEL.subject].isin(subjects_ind)
        df = PREDICTION_DF[ind].reset_index(drop=True).copy()
        df[MODEL.response] = OBS[draw_ind, ...][ind, ...]

        """ Build model """
        config = Config(toml_path=toml_path)
        config.BUILD_DIR = os.path.join(CONFIG.BUILD_DIR, prefix, draw_dir, N_dir, seed_dir)
        model = HierarchicalBayesianModel(config=config)

        """ Load data """
        df, _ = model.load(df=df)

        """ Fit """
        _, posterior_samples = model.run_inference(df=df)

        """ Predict """
        prediction_df = model.make_prediction_dataset(df=df, num_points=100)
        ppd = model.predict(df=prediction_df, posterior_samples=posterior_samples)

        """ Plot """
        model.render_recruitment_curves(df=df, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=ppd)
        model.render_predictive_check(df=df, prediction_df=prediction_df, posterior_predictive=ppd)

        """ Power """
        mu_delta = np.array(posterior_samples["mu_delta"])

        dst = os.path.join(model.build_dir, "mu_delta.npy")
        np.save(dst, mu_delta)

        config, df, prediction_df, _,  = None, None, None, None
        model, posterior_samples = None, None
        ppd  = None
        mu_delta = None

        del config, df, prediction_df, _, model, posterior_samples, ppd, mu_delta
        gc.collect()
        return


    """ Load """
    toml_path = "/home/vishu/repos/hbmep-paper/configs/human/intraoperative/config.toml"
    CONFIG = Config(toml_path=toml_path)
    MODEL = HierarchicalBayesianModel(config=CONFIG)

    src = "/home/vishu/data/hbmep-processed/human/intraoperative/data.csv"
    DF = pd.read_csv(src)

    subset = [
        'scapptio001'
    ]
    ind = ~DF[MODEL.subject].isin(subset)
    DF = DF[ind].reset_index(drop=True).copy()

    DF, ENCODER_DICT = MODEL.load(df=DF)

    dest = os.path.join(MODEL.build_dir, "inference.pkl")
    with open(dest, "rb") as g:
        _, MCMC, POSTERIOR_SAMPLES = pickle.load(g)

    """ Experiment """
    TOTAL_SUBJECTS = 200

    PREDICTION_DF = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[MODEL.subject]) \
        .merge(
            pd.DataFrame(np.arange(0, 2, 1), columns=MODEL.features),
            how="cross"
        ) \
        .merge(
            pd.DataFrame([0, 10], columns=[MODEL.intensity]),
            how="cross"
        )
    PREDICTION_DF = MODEL.make_prediction_dataset(df=PREDICTION_DF, num_points=20)

    POST = {u: v for u, v in POSTERIOR_SAMPLES.items() if u not in MODEL.priors}
    POSTERIOR_PREDICTIVE = MODEL.predict(df=PREDICTION_DF, posterior_samples=POST)

    OBS = np.array(POSTERIOR_PREDICTIVE[site.obs])
    mu_delta_true = POSTERIOR_SAMPLES["mu_delta"]

    prefix = "power-analysis-parallel"
    N_space = [2, 4, 8, 12, 16, 20]

    keys = jax.random.split(MODEL.rng_key, num=2)

    n_draws = 50
    draws_space = \
        jax.random.choice(
            key=keys[0],
            a=np.arange(0, mu_delta_true.shape[0], 1),
            shape=(n_draws,),
            replace=False
        ) \
        .tolist()

    n_repeats = 50
    repeats_space = \
        jax.random.choice(
            key=keys[1],
            a=np.arange(0, n_repeats * 100, 1),
            shape=(n_repeats,),
            replace=False
        ) \
        .tolist()

    parallel = Parallel(n_jobs=4)
    parallel(
        delayed(_process)(N_counter, draw_counter, repeat_counter) \
        for draw_counter in range(7, n_draws) \
        for N_counter in range(len(N_space)) \
        for repeat_counter in range(n_repeats)
    )
