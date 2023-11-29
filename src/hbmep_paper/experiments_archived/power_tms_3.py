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


class Simulation(Baseline):
    LINK = HBM

    def __init__(self, config: Config, mu_a_delta: int, sigma_a_delta: int):
        super(Simulation, self).__init__(config=config)
        self.combination_columns = self.features + [self.subject]
        self.mu_a_delta = mu_a_delta
        self.sigma_a_delta = sigma_a_delta

    def _model(self, subject, features, intensity, response_obs=None):
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_data = intensity.shape[0]
        n_subject = np.unique(subject).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        n_baseline = 1
        n_delta = 1

        global_sigma_b_baseline = numpyro.sample("global_sigma_b_baseline", dist.HalfNormal(100))
        global_sigma_v_baseline = numpyro.sample("global_sigma_v_baseline", dist.HalfNormal(100))

        global_sigma_L_baseline = numpyro.sample("global_sigma_L_baseline", dist.HalfNormal(50))
        global_sigma_H_baseline = numpyro.sample("global_sigma_H_baseline", dist.HalfNormal(500))

        global_sigma_g_1_baseline = numpyro.sample("global_sigma_g_1_baseline", dist.HalfNormal(100))
        global_sigma_g_2_baseline = numpyro.sample("global_sigma_g_2_baseline", dist.HalfNormal(100))

        """ Baseline """
        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_baseline", n_baseline, dim=-2):
                """ Hyper-priors """
                mu_a_baseline = numpyro.sample(
                    "mu_a_baseline",
                    dist.TruncatedNormal(50, 50, low=0)
                )
                sigma_a_baseline = numpyro.sample(
                    "sigma_a_baseline",
                    dist.HalfNormal(50)
                )

                sigma_b_baseline = numpyro.sample("sigma_b_baseline", dist.HalfNormal(global_sigma_b_baseline))
                sigma_v_baseline = numpyro.sample("sigma_v_baseline", dist.HalfNormal(global_sigma_v_baseline))

                sigma_L_baseline = numpyro.sample("sigma_L_baseline", dist.HalfNormal(global_sigma_L_baseline))
                sigma_H_baseline = numpyro.sample("sigma_H_baseline", dist.HalfNormal(global_sigma_H_baseline))

                sigma_g_1_baseline = numpyro.sample("sigma_g_1_baseline", dist.HalfNormal(global_sigma_g_1_baseline))
                sigma_g_2_baseline = numpyro.sample("sigma_g_2_baseline", dist.HalfNormal(global_sigma_g_2_baseline))

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    """ Priors """
                    a_baseline = numpyro.sample(
                        "a_baseline",
                        dist.TruncatedNormal(mu_a_baseline, sigma_a_baseline, low=0)
                    )

                    b_baseline = numpyro.sample("b_baseline", dist.HalfNormal(sigma_b_baseline))
                    v_baseline = numpyro.sample("v_baseline", dist.HalfNormal(sigma_v_baseline))

                    L_baseline = numpyro.sample("L_baseline", dist.HalfNormal(sigma_L_baseline))
                    H_baseline = numpyro.sample("H_baseline", dist.HalfNormal(sigma_H_baseline))

                    g_1_baseline = numpyro.sample("g_1_baseline", dist.Exponential(sigma_g_1_baseline))
                    g_2_baseline = numpyro.sample("g_2_baseline", dist.Exponential(sigma_g_2_baseline))

        """ Delta """
        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_delta", n_delta, dim=-2):
                mu_a_delta = numpyro.deterministic("mu_a_delta", self.mu_a_delta)
                sigma_a_delta = numpyro.deterministic("sigma_a_delta", self.sigma_a_delta)

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    a_delta = numpyro.sample("a_delta", dist.Normal(mu_a_delta, sigma_a_delta))

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_feature0", n_feature0, dim=-2):
                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_baseline, a_baseline + a_delta], axis=1)
                    )

                    b = numpyro.deterministic(
                        site.b,
                        jnp.concatenate([b_baseline, b_baseline], axis=1)
                    )
                    v = numpyro.deterministic(
                        site.v,
                        jnp.concatenate([v_baseline, v_baseline], axis=1)
                    )

                    L = numpyro.deterministic(
                        site.L,
                        jnp.concatenate([L_baseline, L_baseline], axis=1)
                    )
                    H = numpyro.deterministic(
                        site.H,
                        jnp.concatenate([H_baseline, H_baseline], axis=1)
                    )

                    g_1 = numpyro.deterministic(
                        site.g_1,
                        jnp.concatenate([g_1_baseline, g_1_baseline], axis=1)
                    )
                    g_2 = numpyro.deterministic(
                        site.g_2,
                        jnp.concatenate([g_2_baseline, g_2_baseline], axis=1)
                    )

                    # a = numpyro.deterministic(site.a, a_baseline)

                    # b = numpyro.deterministic(site.b, b_baseline)
                    # v = numpyro.deterministic(site.v, v_baseline)

                    # L = numpyro.deterministic(site.L, L_baseline)
                    # H = numpyro.deterministic(site.H, H_baseline)

                    # g_1 = numpyro.deterministic(site.g_1, g_1_baseline)
                    # g_2 = numpyro.deterministic(site.g_2, g_2_baseline)

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

        # """ Penalty """
        # penalty = (jnp.fabs(baseline + delta) - (baseline + delta))
        # numpyro.factor("penalty", -penalty)

        """ Observation """
        with numpyro.plate(site.data, n_data):
            return numpyro.sample(
                site.obs,
                dist.Gamma(concentration=mu * beta, rate=beta).to_event(1),
                obs=response_obs
            )


class HierarchicalBayesianModel(Baseline):
    LINK = HBM

    def __init__(self, config: Config):
        super(HierarchicalBayesianModel, self).__init__(config=config)
        self.combination_columns = self.features + [self.subject]

    def _model(self, subject, features, intensity, response_obs=None):
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_data = intensity.shape[0]
        n_subject = np.unique(subject).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        n_baseline = 1
        n_delta = 1

        global_sigma_b_baseline = numpyro.sample("global_sigma_b_baseline", dist.HalfNormal(100))
        global_sigma_v_baseline = numpyro.sample("global_sigma_v_baseline", dist.HalfNormal(100))

        global_sigma_L_baseline = numpyro.sample("global_sigma_L_baseline", dist.HalfNormal(50))
        global_sigma_H_baseline = numpyro.sample("global_sigma_H_baseline", dist.HalfNormal(500))

        global_sigma_g_1_baseline = numpyro.sample("global_sigma_g_1_baseline", dist.HalfNormal(100))
        global_sigma_g_2_baseline = numpyro.sample("global_sigma_g_2_baseline", dist.HalfNormal(100))

        """ Baseline """
        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_baseline", n_baseline, dim=-2):
                """ Hyper-priors """
                mu_a_baseline = numpyro.sample(
                    "mu_a_baseline",
                    dist.TruncatedNormal(50, 50, low=0)
                )
                sigma_a_baseline = numpyro.sample(
                    "sigma_a_baseline",
                    dist.HalfNormal(50)
                )

                sigma_b_baseline = numpyro.sample("sigma_b_baseline", dist.HalfNormal(global_sigma_b_baseline))
                sigma_v_baseline = numpyro.sample("sigma_v_baseline", dist.HalfNormal(global_sigma_v_baseline))

                sigma_L_baseline = numpyro.sample("sigma_L_baseline", dist.HalfNormal(global_sigma_L_baseline))
                sigma_H_baseline = numpyro.sample("sigma_H_baseline", dist.HalfNormal(global_sigma_H_baseline))

                sigma_g_1_baseline = numpyro.sample("sigma_g_1_baseline", dist.HalfNormal(global_sigma_g_1_baseline))
                sigma_g_2_baseline = numpyro.sample("sigma_g_2_baseline", dist.HalfNormal(global_sigma_g_2_baseline))

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    """ Priors """
                    a_baseline = numpyro.sample(
                        "a_baseline",
                        dist.TruncatedNormal(mu_a_baseline, sigma_a_baseline, low=0)
                    )

                    b_baseline = numpyro.sample("b_baseline", dist.HalfNormal(sigma_b_baseline))
                    v_baseline = numpyro.sample("v_baseline", dist.HalfNormal(sigma_v_baseline))

                    L_baseline = numpyro.sample("L_baseline", dist.HalfNormal(sigma_L_baseline))
                    H_baseline = numpyro.sample("H_baseline", dist.HalfNormal(sigma_H_baseline))

                    g_1_baseline = numpyro.sample("g_1_baseline", dist.Exponential(sigma_g_1_baseline))
                    g_2_baseline = numpyro.sample("g_2_baseline", dist.Exponential(sigma_g_2_baseline))

        """ Delta """
        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_delta", n_delta, dim=-2):
                mu_a_delta = numpyro.sample("mu_a_delta", dist.Normal(0, 100))
                sigma_a_delta = numpyro.sample("sigma_a_delta", dist.HalfNormal(100))

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    a_delta = numpyro.sample("a_delta", dist.Normal(mu_a_delta, sigma_a_delta))

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_feature0", n_feature0, dim=-2):
                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_baseline, a_baseline + a_delta], axis=1)
                    )

                    b = numpyro.deterministic(
                        site.b,
                        jnp.concatenate([b_baseline, b_baseline], axis=1)
                    )
                    v = numpyro.deterministic(
                        site.v,
                        jnp.concatenate([v_baseline, v_baseline], axis=1)
                    )

                    L = numpyro.deterministic(
                        site.L,
                        jnp.concatenate([L_baseline, L_baseline], axis=1)
                    )
                    H = numpyro.deterministic(
                        site.H,
                        jnp.concatenate([H_baseline, H_baseline], axis=1)
                    )

                    g_1 = numpyro.deterministic(
                        site.g_1,
                        jnp.concatenate([g_1_baseline, g_1_baseline], axis=1)
                    )
                    g_2 = numpyro.deterministic(
                        site.g_2,
                        jnp.concatenate([g_2_baseline, g_2_baseline], axis=1)
                    )

                    # a = numpyro.deterministic(site.a, a_baseline)

                    # b = numpyro.deterministic(site.b, b_baseline)
                    # v = numpyro.deterministic(site.v, v_baseline)

                    # L = numpyro.deterministic(site.L, L_baseline)
                    # H = numpyro.deterministic(site.H, H_baseline)

                    # g_1 = numpyro.deterministic(site.g_1, g_1_baseline)
                    # g_2 = numpyro.deterministic(site.g_2, g_2_baseline)

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

        # """ Penalty """
        # penalty = (jnp.fabs(baseline + delta) - (baseline + delta))
        # numpyro.factor("penalty", -penalty)

        """ Observation """
        with numpyro.plate(site.data, n_data):
            return numpyro.sample(
                site.obs,
                dist.Gamma(concentration=mu * beta, rate=beta).to_event(1),
                obs=response_obs
            )


if __name__ == "__main__":

    prefix = "power-analysis-parallel"

    mu_a_delta = 0
    sigma_a_delta = 1

    prefix_2 = f"mu_a_delta_{mu_a_delta}_sigma_a_delta_{sigma_a_delta}"

    toml_path = "/home/vishu/repos/hbmep-paper/configs/human/tms/mixed-effects.toml"

    CONFIG = Config(toml_path=toml_path)

    MODEL = Simulation(config=CONFIG, mu_a_delta=mu_a_delta, sigma_a_delta=sigma_a_delta)

    src = "/home/vishu/data/hbmep-processed/human/tms/data.csv"
    DF = pd.read_csv(src)

    DF[MODEL.features[0]] = 0

    DF[MODEL.response] = DF[MODEL.response] * 1000

    DF, ENCODER_DICT = MODEL.load(df=DF)

    dest = os.path.join(MODEL.build_dir, "inference.pkl")
    with open(dest, "rb") as g:
        _, MCMC, POSTERIOR_SAMPLES = pickle.load(g)

    priors = {
        site.a, "a_baseline",
        site.b, "b_baseline",
        site.v, "v_baseline",
        site.L, "L_baseline",
        site.H, "H_baseline",
        site.g_1, "g_1_baseline",
        site.g_2, "g_2_baseline",
        site.mu, site.beta
    }

    POST = {u: v for u, v in POSTERIOR_SAMPLES.items() if u not in priors}

    """ Experiment """
    TOTAL_SUBJECTS = 200

    PREDICTION_DF = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[MODEL.subject]) \
        .merge(
            pd.DataFrame(np.arange(0, 2, 1), columns=MODEL.features),
            how="cross"
        ) \
        .merge(
            pd.DataFrame([0, 100], columns=[MODEL.intensity]),
            how="cross"
        )
    PREDICTION_DF = MODEL.make_prediction_dataset(df=PREDICTION_DF, num_points=60)

    POSTERIOR_PREDICTIVE = MODEL.predict(df=PREDICTION_DF, posterior_samples=POST)
    OBS = np.array(POSTERIOR_PREDICTIVE[site.obs])

    N_space = [2, 4, 6, 8, 12, 16, 20]
    # N_space = [2, 4, 6, 8, 10, 12]

    keys = jax.random.split(MODEL.rng_key, num=2)

    n_draws = 50
    # n_draws = 2

    draws_space = \
        jax.random.choice(
            key=keys[0],
            a=np.arange(0, CONFIG.MCMC_PARAMS["num_chains"] * CONFIG.MCMC_PARAMS["num_samples"], 1),
            shape=(n_draws,),
            replace=False
        ) \
        .tolist()

    n_repeats = 50
    # n_repeats = 5

    repeats_space = \
        jax.random.choice(
            key=keys[1],
            a=np.arange(0, n_repeats * 100, 1),
            shape=(n_repeats,),
            replace=False
        ) \
        .tolist()


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
        config.BUILD_DIR = os.path.join(CONFIG.BUILD_DIR, prefix, prefix_2, draw_dir, N_dir, seed_dir)
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
        mu_delta = np.array(posterior_samples["mu_a_delta"])

        dst = os.path.join(model.build_dir, "mu_a_delta.npy")
        np.save(dst, mu_delta)

        config, df, prediction_df, _,  = None, None, None, None
        model, posterior_samples = None, None
        ppd  = None
        mu_delta = None

        del config, df, prediction_df, _, model, posterior_samples, ppd, mu_delta
        gc.collect()
        return


    parallel = Parallel(n_jobs=-1)
    parallel(
        delayed(_process)(N_counter, draw_counter, repeat_counter) \
        for draw_counter in range(n_draws) \
        for N_counter in range(len(N_space)) \
        for repeat_counter in range(n_repeats)
    )
