#!/usr/bin/env python
# coding: utf-8
# %%
import os
import pickle
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

import arviz as az
import numpyro

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

str_date = datetime.today().strftime('%Y-%m-%d')

# In[10]:


import numpyro.distributions as dist
from hbmep.model import BaseModel


class LearnPosterior(BaseModel):
    LINK = "learn_posterior"

    def __init__(self, config: Config):
        super(LearnPosterior, self).__init__(config=config)
        # self.combination_columns = self.features + [self.subject]
        self.combination_columns = [self.subject] + self.features

    def fn(self, x, a, b, v, L, ell, H):
        return (
            L
            + jnp.where(
                jnp.less(x, a),
                0.,
                -ell + jnp.true_divide(
                    H + ell,
                    jnp.power(
                        1
                        + jnp.multiply(
                            -1
                            + jnp.power(
                                jnp.true_divide(H + ell, ell),
                                v
                            ),
                            jnp.exp(jnp.multiply(-b, x - a))
                        ),
                        jnp.true_divide(1, v)
                    )
                )
            )
        )

    def _model(self, subject, features, intensity, response_obs=None):
        subject, n_subject = subject
        features, n_features = features
        intensity, n_data = intensity

        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)
        n_feature0 = n_features[0]

        with numpyro.plate(site.n_response, self.n_response):
            """ Global Priors """
            b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(.1))
            v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(.1))

            L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(.1))
            ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(5))
            H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(2))

            g_1_scale_global_scale = numpyro.sample("g_1_scale_global_scale", dist.HalfNormal(5))
            g_2_scale_global_scale = numpyro.sample("g_2_scale_global_scale", dist.HalfNormal(5))

            with numpyro.plate("n_feature0", n_feature0):
                """ Hyper-priors """
                a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(50, 10, low=0))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(20.0))

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

                with numpyro.plate(site.n_subject, n_subject):
                    """ Priors """
                    a = numpyro.sample(
                        "a", dist.TruncatedNormal(a_mean, a_scale, low=0)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    v_raw = numpyro.sample("v_raw", dist.HalfNormal(scale=1))
                    v = numpyro.deterministic(site.v, jnp.multiply(v_scale, v_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                    ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                    ell = numpyro.deterministic("ell", jnp.multiply(ell_scale, ell_raw))

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                    g_1_raw = numpyro.sample("g_1_raw", dist.HalfCauchy(scale=1))
                    g_1 = numpyro.deterministic(site.g_1, jnp.multiply(g_1_scale, g_1_raw))

                    g_2_raw = numpyro.sample("g_2_raw", dist.HalfCauchy(scale=1))
                    g_2 = numpyro.deterministic(site.g_2, jnp.multiply(g_2_scale, g_2_raw))

        """ Outlier Distribution """
        outlier_prob = numpyro.sample("outlier_prob", dist.Uniform(0., .01))
        outlier_scale = numpyro.sample("outlier_scale", dist.HalfNormal(10))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    self.fn(
                        x=intensity,
                        a=a[subject, feature0],
                        b=b[subject, feature0],
                        v=v[subject, feature0],
                        L=L[subject, feature0],
                        ell=ell[subject, feature0],
                        H=H[subject, feature0]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    g_1[subject, feature0] + jnp.true_divide(g_2[subject, feature0], mu)
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


# In[11]:
toml_path = "/home/mcintosh/Local/gitprojects/hbmep-paper/configs/paper/tms/config.toml"

config = Config(toml_path=toml_path)
config.BUILD_DIR = r'/home/mcintosh/Cloud/Research/reports/2023/2023-11-30_paired_recruitment/' + str_date + '_paired'
config.RESPONSE = ["PKPK_APB", "PKPK_ADM"]
config.MCMC_PARAMS["num_warmup"] = 4000
config.MCMC_PARAMS["num_samples"] = 1000
config.FEATURES = ["protocol"]

model = LearnPosterior(config=config)

# In[12]:
src = "/home/mcintosh/Local/temp/test_hbmep/data/proc_2023-11-29_paired.csv"
df = pd.read_csv(src)

src = "/home/mcintosh/Local/temp/test_hbmep/data/proc_2023-11-29_paired.npy"
mat = np.load(src)

subset = ["SCA04", "SCA07"]
# , "SCA11"
ind = df[model.subject].isin(subset)
df = df[ind].reset_index(drop=True).copy()
mat = mat[ind, ...]

ind = df['stim_type'].isin(['TMS'])
df = df[ind].reset_index(drop=True).copy()
mat = mat[ind, ...]

# df = df.astype({'TSCSInt': 'string'})

df, encoder_dict = model.load(df=df)


# In[13]:


model.plot(df=df, encoder_dict=encoder_dict, mep_matrix=mat)


# In[14]:


mcmc, posterior_samples = model.run_inference(df=df)


# In[15]:
_posterior_samples = posterior_samples.copy()
_posterior_samples["outlier_prob"] = _posterior_samples["outlier_prob"] * 0

prediction_df = model.make_prediction_dataset(df=df)
posterior_predictive = model.predict(df=prediction_df, posterior_samples=_posterior_samples)

model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=_posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)


# In[11]:
#
#
# dest = os.path.join(model.build_dir, "inference.pkl")
# with open(dest, "wb") as f:
#     pickle.dump((model, mcmc, posterior_samples), f)
#
# print(dest)
#
#
# # In[12]:
#
#
# dest = os.path.join(model.build_dir, "inference.nc")
# az.to_netcdf(mcmc, dest)
# dest
#
#
# # In[13]:
#
#
# numpyro_data = az.from_numpyro(mcmc)
#
# """ Model evaluation """
# logger.info("Evaluating model ...")
#
# score = az.loo(numpyro_data, var_name=site.obs)
# logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
#
# score = az.waic(numpyro_data, var_name=site.obs)
# logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")
#
#
# # In[15]:
#
#
# type(posterior_samples)
# posterior_samples.keys()
# posterior_samples
#