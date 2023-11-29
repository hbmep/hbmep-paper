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


class HierarchicalBayesianSimulator(Baseline):
    LINK = "hierarchical_bayesian_simulator"

    def __init__(self, config: Config, mu_a_delta, sigma_a_delta):
        super(HierarchicalBayesianSimulator, self).__init__(config=config)
        self.combination_columns = self.features + [self.subject]
        self.mu_a_delta, self.sigma_a_delta = mu_a_delta, sigma_a_delta

    def _model(self, subject, features, intensity, response_obs=None):
        subject, n_subject = subject
        features, n_features = features
        intensity, n_data = intensity

        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_baseline = 1
        n_feature0 = 2
        n_delta = 1

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            global_sigma_b_baseline = numpyro.sample("global_sigma_b_baseline", dist.HalfNormal(100))
            global_sigma_v_baseline = numpyro.sample("global_sigma_v_baseline", dist.HalfNormal(100))

            global_sigma_L_baseline = numpyro.sample("global_sigma_L_baseline", dist.HalfNormal(1))
            global_sigma_l_baseline = numpyro.sample("global_sigma_l_baseline", dist.HalfNormal(100))
            global_sigma_H_baseline = numpyro.sample("global_sigma_H_baseline", dist.HalfNormal(5))

            global_sigma_g_1_baseline = numpyro.sample("global_sigma_g_1_baseline", dist.HalfNormal(100))
            global_sigma_g_2_baseline = numpyro.sample("global_sigma_g_2_baseline", dist.HalfNormal(100))

            global_sigma_p_baseline = numpyro.sample("global_sigma_p_baseline", dist.HalfNormal(100))

            with numpyro.plate("n_baseline", n_baseline, dim=-2):
                """ Hyper-priors """
                mu_a_baseline = numpyro.sample("mu_a_baseline", dist.HalfNormal(scale=5))
                sigma_a_baseline = numpyro.sample("sigma_a_baseline", dist.HalfNormal(scale=1 / 10))

                sigma_b_raw_baseline = numpyro.sample("sigma_b_raw_baseline", dist.HalfNormal(scale=1))
                sigma_b_baseline = numpyro.deterministic("sigma_b_baseline", global_sigma_b_baseline * sigma_b_raw_baseline)

                sigma_v_raw_baseline = numpyro.sample("sigma_v_raw_baseline", dist.HalfNormal(scale=1))
                sigma_v_baseline = numpyro.deterministic("sigma_v_baseline", global_sigma_v_baseline * sigma_v_raw_baseline)

                sigma_L_raw_baseline = numpyro.sample("sigma_L_raw_baseline", dist.HalfNormal(scale=1))
                sigma_L_baseline = numpyro.deterministic("sigma_L_baseline", global_sigma_L_baseline * sigma_L_raw_baseline)

                sigma_l_raw_baseline = numpyro.sample("sigma_l_raw_baseline", dist.HalfNormal(scale=1))
                sigma_l_baseline = numpyro.deterministic("sigma_l_baseline", global_sigma_l_baseline * sigma_l_raw_baseline)

                sigma_H_raw_baseline = numpyro.sample("sigma_H_raw_baseline", dist.HalfNormal(scale=1))
                sigma_H_baseline = numpyro.deterministic("sigma_H_baseline", global_sigma_H_baseline * sigma_H_raw_baseline)

                sigma_g_1_raw_baseline = numpyro.sample("sigma_g_1_raw_baseline", dist.HalfNormal(scale=1))
                sigma_g_1_baseline = numpyro.deterministic("sigma_g_1_baseline", global_sigma_g_1_baseline * sigma_g_1_raw_baseline)

                sigma_g_2_raw_baseline = numpyro.sample("sigma_g_2_raw_baseline", dist.HalfNormal(scale=1))
                sigma_g_2_baseline = numpyro.deterministic("sigma_g_2_baseline", global_sigma_g_2_baseline * sigma_g_2_raw_baseline)

                sigma_p_raw_baseline = numpyro.sample("sigma_p_raw_baseline", dist.HalfNormal(scale=1))
                sigma_p_baseline = numpyro.deterministic("sigma_p_baseline", global_sigma_p_baseline * sigma_p_raw_baseline)

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    """ Priors """
                    a_raw_baseline = numpyro.sample("a_raw_baseline", dist.Gamma(concentration=mu_a_baseline, rate=1))
                    a_baseline = numpyro.deterministic("a_baseline", (1 / sigma_a_baseline) * a_raw_baseline)

                    b_raw_baseline = numpyro.sample("b_raw_baseline", dist.HalfNormal(scale=1))
                    b_baseline = numpyro.deterministic("b_baseline", sigma_b_baseline * b_raw_baseline)

                    v_raw_baseline = numpyro.sample("v_raw_baseline", dist.HalfNormal(scale=1))
                    v_baseline = numpyro.deterministic("v_baseline", sigma_v_baseline * v_raw_baseline)

                    L_raw_baseline = numpyro.sample("L_raw_baseline", dist.HalfNormal(scale=1))
                    L_baseline = numpyro.deterministic("L_baseline", sigma_L_baseline * L_raw_baseline)

                    l_raw_baseline = numpyro.sample("l_raw_baseline", dist.HalfNormal(scale=1))
                    l_baseline = numpyro.deterministic("l_baseline", sigma_l_baseline * l_raw_baseline)

                    H_raw_baseline = numpyro.sample("H_raw_baseline", dist.HalfNormal(scale=1))
                    H_baseline = numpyro.deterministic("H_baseline", sigma_H_baseline * H_raw_baseline)

                    g_1_raw_baseline = numpyro.sample("g_1_raw_baseline", dist.HalfCauchy(scale=1))
                    g_1_baseline = numpyro.deterministic("g_1_baseline", sigma_g_1_baseline * g_1_raw_baseline)

                    g_2_raw_baseline = numpyro.sample("g_2_raw_baseline", dist.HalfCauchy(scale=1))
                    g_2_baseline = numpyro.deterministic("g_2_baseline", sigma_g_2_baseline * g_2_raw_baseline)

                    p_raw_baseline = numpyro.sample("p_raw_baseline", dist.HalfNormal(scale=1))
                    p_baseline = numpyro.deterministic("p_baseline", sigma_p_baseline * p_raw_baseline)

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
                    """ Deterministic """
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
                    l = numpyro.deterministic(
                        "l",
                        jnp.concatenate([l_baseline, l_baseline], axis=1)
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

                    p = numpyro.deterministic(
                        "p",
                        jnp.concatenate([p_baseline, p_baseline], axis=1)
                    )

        """ Penalty """
        penalty = (jnp.fabs(a_baseline + a_delta) - (a_baseline + a_delta))
        numpyro.factor("a_penalty", -penalty)

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate(site.data, n_data, dim=-2):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    L[subject, feature0]
                    + jnp.where(
                        intensity <= a[subject, feature0],
                        0,
                        -l[subject, feature0]
                        + (
                            (H[subject, feature0] + l[subject, feature0])
                            / jnp.power(
                                1
                                + (
                                    (
                                        -1
                                        + jnp.power(
                                            (H[subject, feature0] + l[subject, feature0]) / l[subject, feature0],
                                            v[subject, feature0]
                                        )
                                    )
                                    * jnp.exp(-b[subject, feature0] * (intensity - a[subject, feature0]))
                                ),
                                1 / v[subject, feature0]
                            )
                        )
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    g_1[subject, feature0] + g_2[subject, feature0] * jnp.power(1 / (mu + 1), p[subject, feature0])
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=mu * beta, rate=beta),
                    obs=response_obs
                )


class HierarchicalBayesian(Baseline):
    LINK = "hierarchical_bayesian"

    def __init__(self, config: Config):
        super(HierarchicalBayesian, self).__init__(config=config)
        self.combination_columns = self.features + [self.subject]

    def _model(self, subject, features, intensity, response_obs=None):
        subject, n_subject = subject
        features, n_features = features
        intensity, n_data = intensity

        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_baseline = 1
        n_feature0 = 2
        n_delta = 1

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_baseline", n_baseline, dim=-2):
                """ Hyper-priors """
                mu_a_baseline = numpyro.sample("mu_a_baseline", dist.HalfNormal(scale=5))
                sigma_a_baseline = numpyro.sample("sigma_a_baseline", dist.HalfNormal(scale=1 / 10))

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    """ Priors """
                    a_raw_baseline = numpyro.sample("a_raw_baseline", dist.Gamma(concentration=mu_a_baseline, rate=1))
                    a_baseline = numpyro.deterministic("a_baseline", (1 / sigma_a_baseline) * a_raw_baseline)

        """ Delta """
        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_delta", n_delta, dim=-2):
                mu_a_delta = numpyro.sample("mu_a_delta", dist.Normal(0, 100))
                sigma_a_delta = numpyro.sample("sigma_a_delta", dist.HalfNormal(100))

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    a_delta = numpyro.sample("a_delta", dist.Normal(mu_a_delta, sigma_a_delta))

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            global_sigma_b = numpyro.sample("global_sigma_b", dist.HalfNormal(100))
            global_sigma_v = numpyro.sample("global_sigma_v", dist.HalfNormal(100))

            global_sigma_L = numpyro.sample("global_sigma_L", dist.HalfNormal(1))
            global_sigma_l = numpyro.sample("global_sigma_l", dist.HalfNormal(100))
            global_sigma_H = numpyro.sample("global_sigma_H", dist.HalfNormal(5))

            global_sigma_g_1 = numpyro.sample("global_sigma_g_1", dist.HalfNormal(100))
            global_sigma_g_2 = numpyro.sample("global_sigma_g_2", dist.HalfNormal(100))

            global_sigma_p = numpyro.sample("global_sigma_p", dist.HalfNormal(100))

            with numpyro.plate("n_feature0", n_feature0, dim=-2):
                sigma_b_raw = numpyro.sample("sigma_b_raw", dist.HalfNormal(scale=1))
                sigma_b = numpyro.deterministic("sigma_b", global_sigma_b * sigma_b_raw)

                sigma_v_raw = numpyro.sample("sigma_v_raw", dist.HalfNormal(scale=1))
                sigma_v = numpyro.deterministic("sigma_v", global_sigma_v * sigma_v_raw)

                sigma_L_raw = numpyro.sample("sigma_L_raw", dist.HalfNormal(scale=1))
                sigma_L = numpyro.deterministic("sigma_L", global_sigma_L * sigma_L_raw)

                sigma_l_raw = numpyro.sample("sigma_l_raw", dist.HalfNormal(scale=1))
                sigma_l = numpyro.deterministic("sigma_l", global_sigma_l * sigma_l_raw)

                sigma_H_raw = numpyro.sample("sigma_H_raw", dist.HalfNormal(scale=1))
                sigma_H = numpyro.deterministic("sigma_H", global_sigma_H * sigma_H_raw)

                sigma_g_1_raw = numpyro.sample("sigma_g_1_raw", dist.HalfNormal(scale=1))
                sigma_g_1 = numpyro.deterministic("sigma_g_1", global_sigma_g_1 * sigma_g_1_raw)

                sigma_g_2_raw = numpyro.sample("sigma_g_2_raw", dist.HalfNormal(scale=1))
                sigma_g_2 = numpyro.deterministic("sigma_g_2", global_sigma_g_2 * sigma_g_2_raw)

                sigma_p_raw = numpyro.sample("sigma_p_raw", dist.HalfNormal(scale=1))
                sigma_p = numpyro.deterministic("sigma_p", global_sigma_p * sigma_p_raw)

                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    """ Deterministic """
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_baseline, a_baseline + a_delta], axis=1)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic("b", sigma_b * b_raw)

                    v_raw = numpyro.sample("v_raw", dist.HalfNormal(scale=1))
                    v = numpyro.deterministic("v", sigma_v * v_raw)

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic("L", sigma_L * L_raw)

                    l_raw = numpyro.sample("l_raw", dist.HalfNormal(scale=1))
                    l = numpyro.deterministic("l", sigma_l * l_raw)

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic("H", sigma_H * H_raw)

                    g_1_raw = numpyro.sample("g_1_raw", dist.HalfCauchy(scale=1))
                    g_1 = numpyro.deterministic("g_1", sigma_g_1 * g_1_raw)

                    g_2_raw = numpyro.sample("g_2_raw", dist.HalfCauchy(scale=1))
                    g_2 = numpyro.deterministic("g_2", sigma_g_2 * g_2_raw)

                    p_raw = numpyro.sample("p_raw", dist.HalfNormal(scale=1))
                    p = numpyro.deterministic("p", sigma_p * p_raw)

        """ Penalty """
        penalty = (jnp.fabs(a_baseline + a_delta) - (a_baseline + a_delta))
        numpyro.factor("a_penalty", -penalty)

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate(site.data, n_data, dim=-2):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    L[subject, feature0]
                    + jnp.where(
                        intensity <= a[subject, feature0],
                        0,
                        -l[subject, feature0]
                        + (
                            (H[subject, feature0] + l[subject, feature0])
                            / jnp.power(
                                1
                                + (
                                    (
                                        -1
                                        + jnp.power(
                                            (H[subject, feature0] + l[subject, feature0]) / l[subject, feature0],
                                            v[subject, feature0]
                                        )
                                    )
                                    * jnp.exp(-b[subject, feature0] * (intensity - a[subject, feature0]))
                                ),
                                1 / v[subject, feature0]
                            )
                        )
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    g_1[subject, feature0] + g_2[subject, feature0] * jnp.power(1 / (mu + 1), p[subject, feature0])
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=mu * beta, rate=beta),
                    obs=response_obs
                )


class NonHierarchicalBayesian(Baseline):
    LINK = "non_hierarchical_bayesian"

    def __init__(self, config: Config):
        super(NonHierarchicalBayesian, self).__init__(config=config)
        self.combination_columns = self.features + [self.subject]

    def _model(self, subject, features, intensity, response_obs=None):
        subject, n_subject = subject
        features, n_features = features
        intensity, n_data = intensity

        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_baseline = 1
        n_feature0 = 2
        n_delta = 1

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_baseline", n_baseline, dim=-2):
                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    """ Hyper-priors """
                    mu_a_baseline = numpyro.sample("mu_a_baseline", dist.HalfNormal(scale=5))
                    sigma_a_baseline = numpyro.sample("sigma_a_baseline", dist.HalfNormal(scale=1 / 10))

                    """ Priors """
                    a_raw_baseline = numpyro.sample("a_raw_baseline", dist.Gamma(concentration=mu_a_baseline, rate=1))
                    a_baseline = numpyro.deterministic("a_baseline", (1 / sigma_a_baseline) * a_raw_baseline)

        """ Delta """
        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_delta", n_delta, dim=-2):
                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    mu_a_delta = numpyro.sample("mu_a_delta", dist.Normal(0, 100))
                    sigma_a_delta = numpyro.sample("sigma_a_delta", dist.HalfNormal(100))

                    a_delta = numpyro.sample("a_delta", dist.Normal(mu_a_delta, sigma_a_delta))

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate("n_feature0", n_feature0, dim=-2):
                with numpyro.plate(site.n_subject, n_subject, dim=-3):
                    global_sigma_b = numpyro.sample("global_sigma_b", dist.HalfNormal(100))
                    global_sigma_v = numpyro.sample("global_sigma_v", dist.HalfNormal(100))

                    global_sigma_L = numpyro.sample("global_sigma_L", dist.HalfNormal(1))
                    global_sigma_l = numpyro.sample("global_sigma_l", dist.HalfNormal(100))
                    global_sigma_H = numpyro.sample("global_sigma_H", dist.HalfNormal(5))

                    global_sigma_g_1 = numpyro.sample("global_sigma_g_1", dist.HalfNormal(100))
                    global_sigma_g_2 = numpyro.sample("global_sigma_g_2", dist.HalfNormal(100))

                    global_sigma_p = numpyro.sample("global_sigma_p", dist.HalfNormal(100))

                    sigma_b_raw = numpyro.sample("sigma_b_raw", dist.HalfNormal(scale=1))
                    sigma_b = numpyro.deterministic("sigma_b", global_sigma_b * sigma_b_raw)

                    sigma_v_raw = numpyro.sample("sigma_v_raw", dist.HalfNormal(scale=1))
                    sigma_v = numpyro.deterministic("sigma_v", global_sigma_v * sigma_v_raw)

                    sigma_L_raw = numpyro.sample("sigma_L_raw", dist.HalfNormal(scale=1))
                    sigma_L = numpyro.deterministic("sigma_L", global_sigma_L * sigma_L_raw)

                    sigma_l_raw = numpyro.sample("sigma_l_raw", dist.HalfNormal(scale=1))
                    sigma_l = numpyro.deterministic("sigma_l", global_sigma_l * sigma_l_raw)

                    sigma_H_raw = numpyro.sample("sigma_H_raw", dist.HalfNormal(scale=1))
                    sigma_H = numpyro.deterministic("sigma_H", global_sigma_H * sigma_H_raw)

                    sigma_g_1_raw = numpyro.sample("sigma_g_1_raw", dist.HalfNormal(scale=1))
                    sigma_g_1 = numpyro.deterministic("sigma_g_1", global_sigma_g_1 * sigma_g_1_raw)

                    sigma_g_2_raw = numpyro.sample("sigma_g_2_raw", dist.HalfNormal(scale=1))
                    sigma_g_2 = numpyro.deterministic("sigma_g_2", global_sigma_g_2 * sigma_g_2_raw)

                    sigma_p_raw = numpyro.sample("sigma_p_raw", dist.HalfNormal(scale=1))
                    sigma_p = numpyro.deterministic("sigma_p", global_sigma_p * sigma_p_raw)

                    """ Deterministic """
                    a = numpyro.deterministic(
                        site.a,
                        jnp.concatenate([a_baseline, a_baseline + a_delta], axis=1)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic("b", sigma_b * b_raw)

                    v_raw = numpyro.sample("v_raw", dist.HalfNormal(scale=1))
                    v = numpyro.deterministic("v", sigma_v * v_raw)

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic("L", sigma_L * L_raw)

                    l_raw = numpyro.sample("l_raw", dist.HalfNormal(scale=1))
                    l = numpyro.deterministic("l", sigma_l * l_raw)

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic("H", sigma_H * H_raw)

                    g_1_raw = numpyro.sample("g_1_raw", dist.HalfCauchy(scale=1))
                    g_1 = numpyro.deterministic("g_1", sigma_g_1 * g_1_raw)

                    g_2_raw = numpyro.sample("g_2_raw", dist.HalfCauchy(scale=1))
                    g_2 = numpyro.deterministic("g_2", sigma_g_2 * g_2_raw)

                    p_raw = numpyro.sample("p_raw", dist.HalfNormal(scale=1))
                    p = numpyro.deterministic("p", sigma_p * p_raw)

        """ Penalty """
        penalty = (jnp.fabs(a_baseline + a_delta) - (a_baseline + a_delta))
        numpyro.factor("a_penalty", -penalty)

        with numpyro.plate(site.n_response, self.n_response, dim=-1):
            with numpyro.plate(site.data, n_data, dim=-2):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    L[subject, feature0]
                    + jnp.where(
                        intensity <= a[subject, feature0],
                        0,
                        -l[subject, feature0]
                        + (
                            (H[subject, feature0] + l[subject, feature0])
                            / jnp.power(
                                1
                                + (
                                    (
                                        -1
                                        + jnp.power(
                                            (H[subject, feature0] + l[subject, feature0]) / l[subject, feature0],
                                            v[subject, feature0]
                                        )
                                    )
                                    * jnp.exp(-b[subject, feature0] * (intensity - a[subject, feature0]))
                                ),
                                1 / v[subject, feature0]
                            )
                        )
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    g_1[subject, feature0] + g_2[subject, feature0] * jnp.power(1 / (mu + 1), p[subject, feature0])
                )

                """ Observation """
                numpyro.sample(
                    site.obs,
                    dist.Gamma(concentration=mu * beta, rate=beta),
                    obs=response_obs
                )


if __name__ == "__main__":
    """ Initialize simulator  """
    toml_path = "/home/vishu/repos/hbmep-paper/configs/paper/tms/mixed-effects/simulator/hierarchical_bayesian_simulator.toml"
    mu_a_delta, sigma_a_delta = -1.5, 1
    simulation_prefix = f"mu_a_delta_{mu_a_delta}__sigma_a_delta_{sigma_a_delta}"

    CONFIG = Config(toml_path=toml_path)
    CONFIG.BUILD_DIR = os.path.join(CONFIG.BUILD_DIR, simulation_prefix)

    SIMULATOR = HierarchicalBayesianSimulator(config=CONFIG, mu_a_delta=mu_a_delta, sigma_a_delta=sigma_a_delta)

    """ Simulation """
    TOTAL_SUBJECTS = 1000
    PREDICTION_DF = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[SIMULATOR.subject]) \
        .merge(
            pd.DataFrame(np.arange(0, 2, 1), columns=SIMULATOR.features),
            how="cross"
        ) \
        .merge(
            pd.DataFrame([0, 100], columns=[SIMULATOR.intensity]),
            how="cross"
        )
    PREDICTION_DF = SIMULATOR.make_prediction_dataset(df=PREDICTION_DF, num_points=60)

    """ Load learned posterior """
    src = os.path.join(SIMULATOR.build_dir, "POSTERIOR_PREDICTIVE.pkl")
    with open(src, "rb") as g:
        POSTERIOR_PREDICTIVE, = pickle.load(g)
    OBS = np.array(POSTERIOR_PREDICTIVE[site.obs])

    """ Experiment space """
    experiment_prefix = "sprase_subjects_power"

    TOTAL_DRAWS = OBS.shape[0]
    N_space = [1, 2, 4, 6, 8, 12, 16, 20]
    n_draws = 100
    n_repeats = 50

    keys = jax.random.split(SIMULATOR.rng_key, num=2)
    draws_space = \
        jax.random.choice(
            key=keys[0],
            a=np.arange(0, TOTAL_DRAWS, 1),
            shape=(n_draws,),
            replace=False
        ) \
        .tolist()
    repeats_space = \
        jax.random.choice(
            key=keys[1],
            a=np.arange(0, n_repeats * 100, 1),
            shape=(n_repeats,),
            replace=False
        ) \
        .tolist()


    def _process(N_counter, draw_counter, repeat_counter, m):
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

        if m.LINK == "hierarchical_bayesian":
            ind = PREDICTION_DF[SIMULATOR.subject].isin(subjects_ind)
            df = PREDICTION_DF[ind].reset_index(drop=True).copy()
            df[SIMULATOR.response] = OBS[draw_ind, ...][ind, ...]

            """ Build model """
            config = Config(toml_path=toml_path)
            config.BUILD_DIR = os.path.join(CONFIG.BUILD_DIR, experiment_prefix, m.LINK, draw_dir, N_dir, seed_dir)
            model = m(config=config)

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

            """ Save """
            dest = os.path.join(model.build_dir, "inference.pkl")
            with open(dest, "wb") as f:
                pickle.dump((posterior_samples, ), f)

            config, df, prediction_df, _,  = None, None, None, None
            model, posterior_samples = None, None
            ppd  = None

            del config, df, prediction_df, _, model, posterior_samples, ppd
            gc.collect()

        elif m.LINK == "non_hierarchical_bayesian":
            for subject in subjects_ind:
                subject_dir = f"subject_{subject}"

                ind = PREDICTION_DF[SIMULATOR.subject].isin([subject])
                df = PREDICTION_DF[ind].reset_index(drop=True).copy()
                df[SIMULATOR.response] = OBS[draw_ind, ...][ind, ...]

                """ Build model """
                config = Config(toml_path=toml_path)
                config.BUILD_DIR = os.path.join(CONFIG.BUILD_DIR, experiment_prefix, m.LINK, draw_dir, N_dir, seed_dir, subject_dir)
                model = m(config=config)

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

                """ Save """
                dest = os.path.join(model.build_dir, "inference.pkl")
                with open(dest, "wb") as f:
                    pickle.dump((posterior_samples, ), f)

                config, df, prediction_df, _,  = None, None, None, None
                model, posterior_samples = None, None
                ppd  = None

                del config, df, prediction_df, _, model, posterior_samples, ppd
                gc.collect()

        else:
            raise ValueError("Wrong Model")

        return


    # parallel = Parallel(n_jobs=-1)
    # parallel(
    #     delayed(_process)(N_counter, draw_counter, repeat_counter, m) \
    #     for draw_counter in range(0, 3) \
    #     for N_counter in range(len(N_space)) \
    #     for repeat_counter in range(0, 5) \
    #     for m in [NonHierarchicalBayesianModel]
    # )

    parallel = Parallel(n_jobs=-1)
    parallel(
        delayed(_process)(N_counter, draw_counter, repeat_counter, model) \
        for draw_counter in range(1) \
        for N_counter in range(len(N_space)) \
        for repeat_counter in range(1) \
        for model in [HierarchicalBayesian, NonHierarchicalBayesian]
    )

    # parallel = Parallel(n_jobs=-1)
    # parallel(
    #     delayed(_process)(N_counter, draw_counter, repeat_counter, model) \
    #     for draw_counter in range(n_draws) \
    #     for N_counter in range(len(N_space)) \
    #     for repeat_counter in range(n_repeats) \
    #     for model in [HierarchicalBayesian, NonHierarchicalBayesian]
    # )
