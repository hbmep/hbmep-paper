import os
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

import arviz as az
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model import functional as F
from hbmep.model.utils import Site as site

from learn_posterior import LearnPosterior

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
dest = "/home/vishu/logs/simulate.log"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(dest, mode="w"),
        logging.StreamHandler()
    ],
    force=True
)


class HBSimulator(BaseModel):
    NAME = "hb_simulator"

    def __init__(self, config: Config, a_random_mean, a_random_scale):
        super(HBSimulator, self).__init__(config=config)
        self.a_random_mean = a_random_mean
        self.a_random_scale = a_random_scale

    def _model(self, features, intensity, response_obs=None):
        features, n_features = features
        intensity, n_data = intensity
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)      # Subject
        feature1 = features[1].reshape(-1,)      # Intervention
        n_fixed = 1
        n_random = n_features[1] - 1

        a_random_mean, a_random_scale = self.a_random_mean, self.a_random_scale

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
                # a_random_mean = numpyro.sample("a_random_mean", dist.Normal(0, 50))
                # a_random_scale = numpyro.sample("a_random_scale", dist.HalfNormal(50.0))

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

            g_1_scale_global_scale = numpyro.sample("g_1_scale_global_scale", dist.HalfNormal(5))
            g_2_scale_global_scale = numpyro.sample("g_2_scale_global_scale", dist.HalfNormal(5))

            with numpyro.plate("n_fixed", n_fixed):
                """ Hyper-priors """
                b_scale_raw = numpyro.sample("b_scale_raw", dist.HalfNormal(scale=1))
                b_scale = numpyro.deterministic("b_scale", jnp.multiply(b_scale_global_scale, b_scale_raw))

                v_scale_raw = numpyro.sample("v_scale_raw", dist.HalfNormal(scale=1))
                v_scale = numpyro.deterministic("v_scale", jnp.multiply(v_scale_global_scale, v_scale_raw))

                L_scale_raw = numpyro.sample("L_scale_raw", dist.HalfNormal(scale=1))
                L_scale = numpyro.deterministic("L_scale", jnp.multiply(L_scale_global_scale, L_scale_raw))

                ell_scale_raw = numpyro.sample("ell_scale_raw", dist.HalfNormal(scale=1))
                ell_scale = numpyro.deterministic("ell_scale", jnp.multiply(ell_scale_global_scale, ell_scale_raw))

                H_scale_raw = numpyro.sample("H_scale_raw", dist.HalfNormal(scale=1))
                H_scale = numpyro.deterministic("H_scale", jnp.multiply(H_scale_global_scale, H_scale_raw))

                g_1_scale_raw = numpyro.sample("g_1_scale_raw", dist.HalfNormal(scale=1))
                g_1_scale = numpyro.deterministic("g_1_scale", jnp.multiply(g_1_scale_global_scale, g_1_scale_raw))

                g_2_scale_raw = numpyro.sample("g_2_scale_raw", dist.HalfNormal(scale=1))
                g_2_scale = numpyro.deterministic("g_2_scale", jnp.multiply(g_2_scale_global_scale, g_2_scale_raw))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Fixed Effects (Baseline) priors """
                    b_raw_fixed = numpyro.sample("b_raw_fixed", dist.HalfNormal(scale=1))
                    b_fixed = numpyro.deterministic("b_fixed", jnp.multiply(b_scale, b_raw_fixed))

                    v_raw_fixed = numpyro.sample("v_raw_fixed", dist.HalfNormal(scale=1))
                    v_fixed = numpyro.deterministic("v_fixed", jnp.multiply(v_scale, v_raw_fixed))

                    L_raw_fixed = numpyro.sample("L_raw_fixed", dist.HalfNormal(scale=1))
                    L_fixed = numpyro.deterministic("L_fixed", jnp.multiply(L_scale, L_raw_fixed))

                    ell_raw_fixed = numpyro.sample("ell_raw_fixed", dist.HalfNormal(scale=1))
                    ell_fixed = numpyro.deterministic("ell_fixed", jnp.multiply(ell_scale, ell_raw_fixed))

                    H_raw_fixed = numpyro.sample("H_raw_fixed", dist.HalfNormal(scale=1))
                    H_fixed = numpyro.deterministic("H_fixed", jnp.multiply(H_scale, H_raw_fixed))

                    g_1_raw_fixed = numpyro.sample("g_1_raw", dist.HalfCauchy(scale=1))
                    g_1_fixed = numpyro.deterministic("g_1_fixed", jnp.multiply(g_1_scale, g_1_raw_fixed))

                    g_2_raw_fixed = numpyro.sample("g_2_raw_fixed", dist.HalfCauchy(scale=1))
                    g_2_fixed = numpyro.deterministic("g_2_fixed", jnp.multiply(g_2_scale, g_2_raw_fixed))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_fixed, a_fixed + a_random], axis=1)
                    )
                    b = numpyro.deterministic(
                        site.b,
                        jnp.concatenate([b_fixed, b_fixed], axis=1)
                    )
                    v = numpyro.deterministic(
                        site.v,
                        jnp.concatenate([v_fixed, v_fixed], axis=1)
                    )
                    L = numpyro.deterministic(
                        site.L,
                        jnp.concatenate([L_fixed, L_fixed], axis=1)
                    )
                    ell = numpyro.deterministic(
                        site.ell,
                        jnp.concatenate([ell_fixed, ell_fixed], axis=1)
                    )
                    H = numpyro.deterministic(
                        site.H,
                        jnp.concatenate([H_fixed, H_fixed], axis=1)
                    )
                    g_1 = numpyro.deterministic(
                        site.g_1,
                        jnp.concatenate([g_1_fixed, g_1_fixed], axis=1)
                    )
                    g_2 = numpyro.deterministic(
                        site.g_2,
                        jnp.concatenate([g_2_fixed, g_2_fixed], axis=1)
                    )

        """ Outlier Distribution """
        outlier_prob = numpyro.sample("outlier_prob", dist.Uniform(0., .01))
        outlier_scale = numpyro.sample("outlier_scale", dist.HalfNormal(10))

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
                    g_1[feature0, feature1] + jnp.true_divide(g_2[feature0, feature1], mu)
                )

                q = numpyro.deterministic("q", outlier_prob * jnp.ones((n_data, self.n_response)))
                bg_scale = numpyro.deterministic("bg_scale", outlier_scale * jnp.ones((n_data, self.n_response)))

                mixing_distribution = dist.Categorical(
                    probs=jnp.stack([1 - q, q], axis=-1)
                )
                component_distributions=[
                    dist.Gamma(concentration=jnp.multiply(mu, beta), rate=beta),
                    dist.HalfNormal(scale=bg_scale)
                ]

                """ Mixture """
                Mixture = dist.MixtureGeneral(
                    mixing_distribution=mixing_distribution,
                    component_distributions=component_distributions
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    Mixture,
                    obs=response_obs
                )


def main():
    a_random_mean, a_random_scale = -5, 2

    toml_path = "/home/vishu/repos/hbmep-paper/configs/paper/tms/config.toml"
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "experiments", "sparse-subjects", "hb-simulate-data", f"a_random_mean_{a_random_mean}_a_random_scale_{a_random_scale}")
    config.FEATURES = ["participant", "intervention"]
    config.RESPONSE = ["PKPK_APB"]
    config.MCMC_PARAMS["num_warmup"] = 10000
    config.MCMC_PARAMS["num_samples"] = 4000

    """ Initialize simulator """
    simulator = HBSimulator(config=config, a_random_mean=a_random_mean, a_random_scale=a_random_scale)
    simulator._make_dir(simulator.build_dir)

    """ Load learn posterior """
    src = "/home/vishu/repos/hbmep-paper/reports/paper/tms/experiments/sparse-subjects/learn-posterior/inference.pkl"
    with open(src, "rb") as g:
        _, _, posterior_samples_learnt = pickle.load(g)
    """ Create template dataframe for simulation """
    N_SUBJECTS_TO_SIMULATE = 12
    simulation_df = \
        pd.DataFrame(np.arange(0, N_SUBJECTS_TO_SIMULATE, 1), columns=[simulator.features[0]]) \
        .merge(
            pd.DataFrame(np.arange(0, 2, 1), columns=simulator.features[1:]),
            how="cross"
        ) \
        .merge(
            pd.DataFrame([0, 90], columns=[simulator.intensity]),
            how="cross"
        )
    simulation_df = simulator.make_prediction_dataset(
        df=simulation_df,
        min_intensity=0,
        max_intensity=100,
        num=48
    )
    logger.info(f"Simulation dataframe: {simulation_df.shape}")

    """ Simulate data """
    priors = {
        "global-priors": [
            "b_scale_global_scale",
            "v_scale_global_scale",
            "L_scale_global_scale",
            "ell_scale_global_scale",
            "H_scale_global_scale",
            "g_1_scale_global_scale",
            "g_2_scale_global_scale"
        ],
        "hyper-priors": [
            "b_scale_raw",
            "v_scale_raw",
            "L_scale_raw",
            "ell_scale_raw",
            "H_scale_raw",
            "g_1_scale_raw",
            "g_2_scale_raw",

            "b_scale",
            "v_scale",
            "L_scale",
            "ell_scale",
            "H_scale",
            "g_1_scale",
            "g_2_scale"
        ],
        "baseline-priors": [
            "a_fixed_mean",
            "a_fixed_scale"
        ]
    }
    sites_to_use_for_simulation = ["outlier_prob"]
    sites_to_use_for_simulation += priors["global-priors"]
    sites_to_use_for_simulation += priors["hyper-priors"]
    sites_to_use_for_simulation += priors["baseline-priors"]
    posterior_samples_learnt = {
        k: v for k, v in posterior_samples_learnt.items() if k in sites_to_use_for_simulation
    }
    # Turn off outlier distribution
    posterior_samples_learnt["outlier_prob"] = 0 * posterior_samples_learnt["outlier_prob"]
    # Simulate
    simulation_posterior_predictive = \
        simulator.predict(df=simulation_df, posterior_samples=posterior_samples_learnt)

    """ Filter valid draws """
    a = simulation_posterior_predictive[site.a]
    b = simulation_posterior_predictive[site.b]
    H = simulation_posterior_predictive[site.H]
    ind = ((a > 10) & (a < 70) & (b > .05) & (H > .1)).all(axis=(1, 2, 3))
    n_valid_draws = ind.sum()
    logger.info(f"No. of valid simulations: {n_valid_draws}")
    simulation_posterior_predictive = {
        k: v[ind, ...] for k, v in simulation_posterior_predictive.items()
    }

    """ Shuffle draws """
    keys = jax.random.split(simulator.rng_key, num=3)
    ind = jax.random.permutation(key=keys[0], x=jnp.arange(0, n_valid_draws, 1))
    simulation_posterior_predictive = {
        k: v[ind, ...] for k, v in simulation_posterior_predictive.items()
    }

    """ Positive response constraint """
    sim_obs = simulation_posterior_predictive[site.obs]
    ind = (sim_obs > 0).all(axis=(1, 2))
    logger.info(f"valid_draws: {ind.shape}")
    logger.info(f"No. of valid simulations after accounting for positive response: {ind.sum()}")
    simulation_posterior_predictive = {
        k: v[ind, ...] for k, v in simulation_posterior_predictive.items()
    }

    """ Save simulation dataframe and posterior predictive """
    dest = os.path.join(simulator.build_dir, "simulation_df.csv")
    simulation_df.to_csv(dest, index=False)
    logger.info(f"Saved simulation dataframe to {dest}")

    dest = os.path.join(simulator.build_dir, "simulation_posterior_predictive.pkl")
    with open(dest, "wb") as f:
        pickle.dump((simulator, simulation_posterior_predictive), f)
    logger.info(f"Saved simulation posterior predictive to {dest}")

    # """ Plot """
    # N_DRAWS_TO_PLOT = 50
    # temp_ppd = {
    #     k: v[:N_DRAWS_TO_PLOT, ...] for k, v in simulation_posterior_predictive.items()
    # }
    # assert temp_ppd[site.obs].shape[0] == N_DRAWS_TO_PLOT

    # temp_df = simulation_df.copy()
    # temp_ppd = {
    #     k: v.swapaxes(0, -1) for k, v in temp_ppd.items()
    # }
    # obs = temp_ppd[site.obs]
    # response = [simulator.response[0] + f"_{i}" for i in range(N_DRAWS_TO_PLOT)]
    # temp_df[response] = obs[0, ...]
    # simulator.render_recruitment_curves(
    #     df=temp_df,
    #     response=response,
    #     response_colors = plt.cm.rainbow(np.linspace(0, 1, N_DRAWS_TO_PLOT)),
    #     prediction_df=temp_df,
    #     posterior_predictive=temp_ppd,
    #     posterior_samples=temp_ppd
    # )


if __name__ == "__main__":
    main()
