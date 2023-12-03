#!/usr/bin/env python
# coding: utf-8
# %%
import os
import pickle
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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

str_date = datetime.today().strftime('%Y-%m-%dT%H%M')

# In[10]:
import numpyro.distributions as dist
from hbmep.model import BaseModel


class LearnPosterior(BaseModel):
    LINK = "learn_posterior"

    def __init__(self, config: Config):
        super(LearnPosterior, self).__init__(config=config)
        self.combination_columns = self.features + [self.subject]
        # self.combination_columns = [self.subject] + self.features

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
        a_low = 0

        """ Global Priors """
        a_scale_global_scale = numpyro.sample("a_scale_global_scale", dist.HalfNormal(100))
        a_mean_global_scale = numpyro.sample("a_mean_global_scale", dist.HalfNormal(100))
        a_mean_global_mean = numpyro.sample("a_mean_global_mean", dist.TruncatedNormal(50, 50, low=a_low))

        b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(100))
        v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(100))

        L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(10))
        ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(100))
        H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(10))

        g_1_scale_global_scale = numpyro.sample("g_1_scale_global_scale", dist.HalfNormal(5))
        g_2_scale_global_scale = numpyro.sample("g_2_scale_global_scale", dist.HalfNormal(5))

        """ Outlier Distribution """
        outlier_prob = numpyro.sample("outlier_prob", dist.Uniform(0., .005))
        outlier_scale = numpyro.sample("outlier_scale", dist.HalfNormal(10))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_subject, n_subject):
                """ Hyper-priors """
                a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(a_mean_global_mean, a_mean_global_scale, low=a_low))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(a_scale_global_scale))

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

                with numpyro.plate("n_feature0", n_feature0):
                    """ Priors """
                    a = numpyro.sample(
                        "a", dist.TruncatedNormal(a_mean, a_scale, low=a_low)
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

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    self.fn(
                        x=intensity,
                        a=a[feature0, subject],
                        b=b[feature0, subject],
                        v=v[feature0, subject],
                        L=L[feature0, subject],
                        ell=ell[feature0, subject],
                        H=H[feature0, subject]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    g_1[feature0, subject] + jnp.true_divide(g_2[feature0, subject], mu)
                )

                main_component = dist.Gamma(concentration=jnp.multiply(mu, beta), rate=beta)
                q = numpyro.deterministic("q", outlier_prob * jnp.ones((n_data, self.n_response)))
                bg_scale = numpyro.deterministic("bg_scale", outlier_scale * jnp.ones((n_data, self.n_response)))

                mixing_distribution = dist.Categorical(
                    probs=jnp.stack([1 - q, q], axis=-1)
                )
                component_distributions=[
                    main_component,
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
stim_type = 'TMS'
# stim_type = 'TSCS'
if stim_type == 'TMS':
    stim_type_alt = 'TMS'
elif stim_type == 'TSCS':
    stim_type_alt = 'TSS'
toml_path = "/home/mcintosh/Local/gitprojects/hbmep-paper/configs/paper/tms/config.toml"
config = Config(toml_path=toml_path)
config.BUILD_DIR = r'/home/mcintosh/Cloud/Research/reports/2023/2023-11-30_paired_recruitment/' + str_date + '_' + stim_type + '_paired'
# config.RESPONSE = ["AUC_APB", "AUC_ADM"]
config.MCMC_PARAMS["num_warmup"] = 5000
config.MCMC_PARAMS["num_samples"] = 2000
config.FEATURES = ["protocol"]
config.INTENSITY = stim_type + 'Int'

model = LearnPosterior(config=config)

# In[12]:
src = "/home/mcintosh/Local/temp/test_hbmep/data/proc_2023-11-29_paired.csv"
df = pd.read_csv(src)

src = "/home/mcintosh/Local/temp/test_hbmep/data/proc_2023-11-29_paired.npy"
mat = np.load(src)

subset = ["SCA04", "SCA07", "SCA11"]
#
ind = df[model.subject].isin(subset)
df = df[ind].reset_index(drop=True).copy()
mat = mat[ind, ...]

ind = df['stim_type'].isin([stim_type_alt])
df = df[ind].reset_index(drop=True).copy()
mat = mat[ind, ...]

# df = df.astype({'TSCSInt': 'string'})

df, encoder_dict = model.load(df=df)

orderby = lambda x: (x[1], x[0])

# In[13]:
model.plot(df=df, encoder_dict=encoder_dict, mep_matrix=mat)

# In[14]:
mcmc, posterior_samples = model.run_inference(df=df)

# %%
dest = os.path.join(model.build_dir, "inference.pkl")
print(dest)
if os.path.isfile(dest):
    with open(dest, "rb") as g:
        model, mcmc, posterior_samples = pickle.load(g)
else:
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)

    dest_nc = os.path.join(model.build_dir, "inference.nc")
    az.to_netcdf(mcmc, dest_nc)

# In[15]:
_posterior_samples = posterior_samples.copy()
_posterior_samples["outlier_prob"] = _posterior_samples["outlier_prob"] * 0

# prediction_df = model.make_prediction_dataset(df=df)
prediction_df = model.make_prediction_dataset(df=df, min_intensity=0, max_intensity=100)
posterior_predictive = model.predict(df=prediction_df, posterior_samples=_posterior_samples)

model.render_recruitment_curves(df=df, encoder_dict=encoder_dict,
                                posterior_samples=_posterior_samples,
                                prediction_df=prediction_df,
                                posterior_predictive=posterior_predictive,
                                orderby=orderby)

model.render_predictive_check(df=df, encoder_dict=encoder_dict,
                              prediction_df=prediction_df,
                              posterior_predictive=posterior_predictive,
                              orderby=orderby)

# %%
prediction_df = model.make_prediction_dataset(df=df)

df_template = prediction_df.copy()
ind1 = df_template[model.subject].isin([0])
ind2 = df_template[model.features[0]].isin([0])  # the 0 index on list is because it is a list
df_template = df_template[ind1 & ind2]

rows = 3
cols = 3
pp = [[None for _ in range(cols)] for _ in range(rows)]
for p in range(rows):
    for f in range(cols):
        df_local = df_template.copy()
        df_local[model.subject] = p
        df_local[model.features[0]] = f
        pp[p][f] = model.predict(df=df_local, posterior_samples=_posterior_samples)

# %%
n_muscles = 6
conditions = ['Sub-tSCS', 'Supra-tSCS', 'Normal']
# I need a good way of linking back to the conditions in the data
colors = sns.color_palette('colorblind')

fig, axs = plt.subplots(rows, n_muscles, figsize=(15, 10))
for ix_p in range(rows):
    for ix_muscle in range(n_muscles):
        ax = axs[ix_p, ix_muscle]
        for ix_cond in range(2):
            first_column_active = pp[ix_p][ix_cond][site.mu][:, :, ix_muscle][:, 0].reshape(-1, 1)
            first_column_base = pp[ix_p][2][site.mu][:, :, ix_muscle][:, 0].reshape(-1, 1)
            Y = ((pp[ix_p][ix_cond][site.mu][:, :, ix_muscle] + first_column_base) /
                 (first_column_active + pp[ix_p][2][site.mu][:, :, ix_muscle]))

            Y = (Y - 1) * 100
            # X = X/
            x = df_template[config.INTENSITY].values
            y = np.mean(Y, 0)
            y1 = np.percentile(Y, 2.5, axis=0)
            y2 = np.percentile(Y, 97.5, axis=0)
            ax.plot(x, y, color=colors[ix_cond], label=conditions[ix_cond])
            ax.fill_between(x, y1, y2, color=colors[ix_cond], alpha=0.3)

            if ix_p == 0 and ix_muscle == 0:
                ax.legend()
                ax.set_ylabel('% Fac. (Paired/Brain-only + Spine-only)')
            if ix_p == 0:
                ax.set_title(config.RESPONSE[ix_muscle].split('_')[1])
            if ix_muscle == 0:
                ax.set_xlabel(config.INTENSITY + ' Intensity')
            ax.set_xlim([0, 70])
plt.show()
fig.savefig(Path(model.build_dir) / "norm_REC.svg", format='svg')
fig.savefig(Path(model.build_dir) / "norm_REC.png", format='png')

# %%
fig, axs = plt.subplots(rows, n_muscles, figsize=(15, 10))
for ix_p in range(rows):
    for ix_muscle in range(n_muscles):
        ax = axs[ix_p, ix_muscle]
        for ix_cond in range(3):
            x = df_template[config.INTENSITY].values
            Y = pp[ix_p][ix_cond][site.mu][:, :, ix_muscle]
            y = np.mean(Y, 0)
            y1 = np.percentile(Y, 2.5, axis=0)
            y2 = np.percentile(Y, 97.5, axis=0)
            ax.plot(x, y, color=colors[ix_cond], label=conditions[ix_cond])
            ax.fill_between(x, y1, y2, color=colors[ix_cond], alpha=0.3)

            df_local = df.copy()
            ind1 = df_local[model.subject].isin([ix_p])
            ind2 = df_local[model.features[0]].isin([ix_cond])  # the 0 index on list is because it is a list
            df_local = df_local[ind1 & ind2]
            x = df_local[config.INTENSITY].values
            y = df_local[config.RESPONSE[ix_muscle]].values
            ax.plot(x, y,
                    color=colors[ix_cond], marker='o', markeredgecolor='w',
                    markerfacecolor=colors[ix_cond], linestyle='None',
                    markeredgewidth=1, markersize=4)

            if ix_p == 0 and ix_muscle == 0:
                ax.legend()
            if ix_p == 0:
                ax.set_title(config.RESPONSE[ix_muscle].split('_')[1])
            if ix_muscle == 0:
                ax.set_ylabel('AUC (uVs)')
                ax.set_xlabel(config.INTENSITY + ' Intensity (%)')
            ax.set_xlim([0, 70])


plt.show()
fig.savefig(Path(model.build_dir) / "REC.svg", format='svg')
fig.savefig(Path(model.build_dir) / "REC.png", format='png')

# In[13]:
# numpyro_data = az.from_numpyro(mcmc)
