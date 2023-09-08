import os
import gc
from pathlib import Path
import multiprocessing
from tqdm import tqdm

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import Baseline
from hbmep.model.utils import Site as site
from hbmep.utils.constants import RECTIFIED_LOGISTIC

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()
gc.disable()

class RectifiedLogistic(Baseline):
    LINK = RECTIFIED_LOGISTIC

    def __init__(self, config: Config):
        super(RectifiedLogistic, self).__init__(config=config)

    def _model(self, subject, features, intensity, response_obs=None):
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0, ...].reshape(-1,)

        n_data = intensity.shape[0]
        n_subject = np.unique(subject).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate(site.n_subject, n_subject, dim=-2):
                """ Hyper-priors """
                mu_a = numpyro.sample(
                    site.mu_a,
                    dist.TruncatedNormal(150, 50, low=0)
                )
                sigma_a = numpyro.sample(site.sigma_a, dist.HalfNormal(50))

                sigma_b = numpyro.sample(site.sigma_b, dist.HalfNormal(0.1))

                sigma_L = numpyro.sample(site.sigma_L, dist.HalfNormal(0.05))
                sigma_H = numpyro.sample(site.sigma_H, dist.HalfNormal(5))
                sigma_v = numpyro.sample(site.sigma_v, dist.HalfNormal(10))

                with numpyro.plate("n_feature0", n_feature0, dim=-3):
                    """ Priors """
                    a = numpyro.sample(
                        site.a,
                        dist.TruncatedNormal(mu_a, sigma_a, low=0)
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
            L[feature0, subject]
            + jnp.maximum(
                0,
                -1
                + (H[feature0, subject] + 1)
                / jnp.power(
                    1
                    + (jnp.power(1 + H[feature0, subject], v[feature0, subject]) - 1)
                    * jnp.exp(-b[feature0, subject] * (intensity - a[feature0, subject])),
                    1 / v[feature0, subject]
                )
            )
        )
        beta = numpyro.deterministic(
            site.beta,
            g_1[feature0, subject] + g_2[feature0, subject] * (1 / mu)
        )

        """ Observation """
        with numpyro.plate(site.data, n_data):
            return numpyro.sample(
                site.obs,
                dist.Gamma(concentration=mu * beta, rate=beta).to_event(1),
                obs=response_obs
            )

# class RectifiedLogistic(Baseline):
#     LINK = RECTIFIED_LOGISTIC

#     def __init__(self, config: Config):
#         super(RectifiedLogistic, self).__init__(config=config)

#     def _model(self, subject, features, intensity, response_obs=None):
#         intensity = intensity.reshape(-1, 1)
#         intensity = np.tile(intensity, (1, self.n_response))

#         feature0 = features[0, ...].reshape(-1,)
#         feature1 = features[1, ...].reshape(-1,)

#         n_data = intensity.shape[0]
#         n_subject = np.unique(subject).shape[0]
#         n_feature0 = np.unique(feature0).shape[0]
#         n_feature1 = np.unique(feature1).shape[0]

#         with numpyro.plate(site.n_response, self.n_response, dim=-1):
#             with numpyro.plate(site.n_subject, n_subject, dim=-2):
#                 """ Hyper-priors """
#                 mu_a = numpyro.sample(
#                     site.mu_a,
#                     dist.TruncatedNormal(150, 50, low=0)
#                 )
#                 sigma_a = numpyro.sample(site.sigma_a, dist.HalfNormal(50))

#                 sigma_b = numpyro.sample(site.sigma_b, dist.HalfNormal(0.1))

#                 sigma_L = numpyro.sample(site.sigma_L, dist.HalfNormal(0.05))
#                 sigma_H = numpyro.sample(site.sigma_H, dist.HalfNormal(5))
#                 sigma_v = numpyro.sample(site.sigma_v, dist.HalfNormal(10))

#                 with numpyro.plate("n_feature0", n_feature0, dim=-3):
#                     with numpyro.plate("n_feature1", n_feature1, dim=-4):
#                         """ Priors """
#                         a = numpyro.sample(
#                             site.a,
#                             dist.TruncatedNormal(mu_a, sigma_a, low=0)
#                         )
#                         b = numpyro.sample(site.b, dist.HalfNormal(sigma_b))

#                         L = numpyro.sample(site.L, dist.HalfNormal(sigma_L))
#                         H = numpyro.sample(site.H, dist.HalfNormal(sigma_H))
#                         v = numpyro.sample(site.v, dist.HalfNormal(sigma_v))

#                         g_1 = numpyro.sample(site.g_1, dist.Exponential(0.01))
#                         g_2 = numpyro.sample(site.g_2, dist.Exponential(0.01))

#         """ Model """
#         mu = numpyro.deterministic(
#             site.mu,
#             L[feature1, feature0, subject]
#             + jnp.maximum(
#                 0,
#                 -1
#                 + (H[feature1, feature0, subject] + 1)
#                 / jnp.power(
#                     1
#                     + (jnp.power(1 + H[feature1, feature0, subject], v[feature1, feature0, subject]) - 1)
#                     * jnp.exp(-b[feature1, feature0, subject] * (intensity - a[feature1, feature0, subject])),
#                     1 / v[feature1, feature0, subject]
#                 )
#             )
#         )
#         beta = numpyro.deterministic(
#             site.beta,
#             g_1[feature1, feature0, subject] + g_2[feature1, feature0, subject] * (1 / mu)
#         )

#         """ Observation """
#         with numpyro.plate(site.data, n_data):
#             return numpyro.sample(
#                 site.obs,
#                 dist.Gamma(concentration=mu * beta, rate=beta).to_event(1),
#                 obs=response_obs
#             )


def run():
    # config.BUILD_DIR = os.path.join(BUILD_DIR, "iterative", f"{c[0]}", f"{c[1]}", f"{c[2]}")
    config.BUILD_DIR = os.path.join(BUILD_DIR, "iterative", f"{c[0]}", f"{c[1]}")
    ind = \
        df[[config.SUBJECT] + config.FEATURES] \
        .apply(tuple, axis=1) \
        .isin([c]) \
        .copy()
    temp_df = df[ind].reset_index(drop=True).copy(deep=False)

    model = RectifiedLogistic(config=config)

    """ Process """
    temp_df, encoder_dict = model.load(df=temp_df)
    mcmc, posterior_samples = model.run_inference(df=temp_df)

    model.render_recruitment_curves(df=temp_df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)
    model.render_predictive_check(df=temp_df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)
    model.save(mcmc=mcmc)

    del ind, temp_df, model, encoder_dict, mcmc, posterior_samples
    collected = gc.collect()
    print(f"Collected {collected}")


if __name__ == "__main__":
    """ Iter J_RCML_000 """
    toml_path = "/home/vishu/repos/hbmep-paper/configs/circ/triceps.toml"
    config = Config(toml_path=toml_path)
    BUILD_DIR = config.BUILD_DIR

    df = pd.read_csv(config.CSV_PATH)
    ind = df.pulse_amplitude.isin([0])
    df = df[~ind].reset_index(drop=True).copy()

    combinations = \
        df[[config.SUBJECT] + config.FEATURES] \
        .apply(tuple, axis=1) \
        .unique() \
        .tolist()
    combinations = sorted(combinations)

    for c in tqdm(combinations):
        run()

    # """ Iter L_SHIE """
    # toml_path = "/home/vishu/repos/hbmep-paper/configs/shie/triceps.toml"
    # config = Config(toml_path=toml_path)
    # BUILD_DIR = config.BUILD_DIR

    # df = pd.read_csv(config.CSV_PATH)
    # ind = df.pulse_amplitude.isin([0])
    # df = df[~ind].reset_index(drop=True).copy()

    # combinations = \
    #     df[[config.SUBJECT] + config.FEATURES] \
    #     .apply(tuple, axis=1) \
    #     .unique() \
    #     .tolist()
    # combinations = sorted(combinations)

    # for c in tqdm(combinations):
    #     run()


    # """ Iter J_RCML_000 """
    # toml_path = "/home/vishu/repos/hbmep-paper/configs/rcml/triceps.toml"
    # config = Config(toml_path=toml_path)
    # BUILD_DIR = config.BUILD_DIR

    # df = pd.read_csv(config.CSV_PATH)
    # ind = df.pulse_amplitude.isin([0])
    # df = df[~ind].reset_index(drop=True).copy()

    # combinations = \
    #     df[[config.SUBJECT] + config.FEATURES] \
    #     .apply(tuple, axis=1) \
    #     .unique() \
    #     .tolist()
    # combinations = sorted(combinations)

    # for c in tqdm(combinations):
    #     run()

    # """ Iter SMA_LAR """
    # toml_path = "/home/vishu/repos/hbmep-paper/configs/small-large/triceps.toml"
    # config = Config(toml_path=toml_path)
    # BUILD_DIR = config.BUILD_DIR

    # df = pd.read_csv(config.CSV_PATH)
    # ind = df.pulse_amplitude.isin([0])
    # df = df[~ind].reset_index(drop=True).copy()

    # combinations = \
    #     df[[config.SUBJECT] + config.FEATURES] \
    #     .apply(tuple, axis=1) \
    #     .unique() \
    #     .tolist()
    # combinations = sorted(combinations)

    # for c in tqdm(combinations):
    #     run()

    # """ Iter SHAP """
    # toml_path = "/home/vishu/repos/hbmep-paper/configs/shap/fcr.toml"
    # config = Config(toml_path=toml_path)
    # BUILD_DIR = config.BUILD_DIR

    # df = pd.read_csv(config.CSV_PATH)
    # ind = df.pulse_amplitude.isin([0])
    # df = df[~ind].reset_index(drop=True).copy()

    # combinations = \
    #     df[[config.SUBJECT] + config.FEATURES] \
    #     .apply(tuple, axis=1) \
    #     .unique() \
    #     .tolist()
    # combinations = sorted(combinations)

    # for c in tqdm(combinations):
    #     config.BUILD_DIR = os.path.join(BUILD_DIR, "iterative", f"{c[0]}", f"{c[1]}", f"{c[2]}")
    #     ind = \
    #         df[[config.SUBJECT] + config.FEATURES] \
    #         .apply(tuple, axis=1) \
    #         .isin([c]) \
    #         .copy()
    #     temp_df = df[ind].reset_index(drop=True).copy()

    #     model = RectifiedLogistic(config=config)

    #     """ Process """
    #     temp_df, encoder_dict = model.load(df=temp_df)
    #     mcmc, posterior_samples = model.run_inference(df=temp_df)

    #     model.render_recruitment_curves(df=temp_df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)
    #     model.render_predictive_check(df=temp_df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)
    #     model.save(mcmc=mcmc)

    # """ Proper """
    # toml_path = "/home/vishu/repos/hbmep-paper/configs/shap/ecr.toml"
    # config = Config(toml_path=toml_path)

    # model = RectifiedLogistic(config=config)

    # df = pd.read_csv(model.csv_path)

    # ind = df.pulse_amplitude.isin([0])
    # df = df[~ind].reset_index(drop=True).copy()

    # """ Process """
    # df, encoder_dict = model.load(df=df)
    # mcmc, posterior_samples = model.run_inference(df=df)

    # model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)
    # model.render_predictive_check(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)
    # model.save(mcmc=mcmc)
