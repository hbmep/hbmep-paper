#!/usr/bin/env python
# coding: utf-8
#
# %%
import os
import pickle
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

import arviz as az
import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jax import grad, vmap

import toml

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.model import BaseModel
from hbmep.nn import functional as F

# In[10]:
class LearnPosterior(BaseModel):
    LINK = "learn_posterior"

    def __init__(self, config: Config):
        super(LearnPosterior, self).__init__(config=config)
        # self.combination_columns = self.features + [self.subject]
        # self.combination_columns = [self.subject] + self.features

    # def gradient_fn(self, x, **kwargs):
    #     grad = jax.grad(F, argnums=0)
    #     for _ in range(len(x.shape)):
    #         grad = jax.vmap(grad)
    #     return grad(x, **kwargs)

    def _model(self, intensity, features, response_obs=None):
        n_data = intensity.shape[0]
        n_features = np.max(features, axis=0) + 1
        feature0 = features[..., 0]
        feature1 = features[..., 1]

        a_low = 0

        """ Global Priors """
        a_scale_global_scale = numpyro.sample("a_scale_global_scale", dist.HalfNormal(100))
        a_mean_global_scale = numpyro.sample("a_mean_global_scale", dist.HalfNormal(100))
        a_mean_global_mean = numpyro.sample("a_mean_global_mean", dist.TruncatedNormal(50, 50, low=a_low))

        b_scale_global_scale = numpyro.sample("b_scale_global_scale", dist.HalfNormal(100))
        v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(100))

        L_scale_global_scale = numpyro.sample("L_scale_global_scale", dist.HalfNormal(10))
        # ell_scale_global_scale = numpyro.sample("ell_scale_global_scale", dist.HalfNormal(100))
        H_scale_global_scale = numpyro.sample("H_scale_global_scale", dist.HalfNormal(20))

        c_1_scale_global_scale = numpyro.sample("c_1_scale_global_scale", dist.HalfNormal(5))
        c_2_scale_global_scale = numpyro.sample("c_2_scale_global_scale", dist.HalfNormal(5))

        """ Outlier Distribution """
        outlier_prob = numpyro.sample("outlier_prob", dist.Uniform(0., .01))
        outlier_scale = numpyro.sample("outlier_scale", dist.HalfNormal(20))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate("ell_subject", 1):
                with numpyro.plate("ell_feature", 1):
                    ell_baseline = numpyro.sample("ell_baseline", dist.HalfNormal(100))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_features[1], n_features[1]):
                """ Hyper-priors """
                a_mean = numpyro.sample("a_mean", dist.TruncatedNormal(a_mean_global_mean, a_mean_global_scale, low=a_low))
                a_scale = numpyro.sample("a_scale", dist.HalfNormal(a_scale_global_scale))

                b_scale_raw = numpyro.sample("b_scale_raw", dist.HalfNormal(scale=1))
                b_scale = numpyro.deterministic("b_scale", jnp.multiply(b_scale_global_scale, b_scale_raw))

                v_scale_raw = numpyro.sample("v_scale_raw", dist.HalfNormal(scale=1))
                v_scale = numpyro.deterministic("v_scale", jnp.multiply(v_scale_global_scale, v_scale_raw))

                L_scale_raw = numpyro.sample("L_scale_raw", dist.HalfNormal(scale=1))
                L_scale = numpyro.deterministic("L_scale", jnp.multiply(L_scale_global_scale, L_scale_raw))

                # ell_scale_raw = numpyro.sample("ell_scale_raw", dist.HalfNormal(scale=1))
                # ell_scale = numpyro.deterministic("ell_scale", jnp.multiply(ell_scale_global_scale, ell_scale_raw))

                H_scale_raw = numpyro.sample("H_scale_raw", dist.HalfNormal(scale=1))
                H_scale = numpyro.deterministic("H_scale", jnp.multiply(H_scale_global_scale, H_scale_raw))

                c_1_scale_raw = numpyro.sample("c_1_scale_raw", dist.HalfNormal(scale=1))
                c_1_scale = numpyro.deterministic("c_1_scale", jnp.multiply(c_1_scale_global_scale, c_1_scale_raw))

                c_2_scale_raw = numpyro.sample("c_2_scale_raw", dist.HalfNormal(scale=1))
                c_2_scale = numpyro.deterministic("c_2_scale", jnp.multiply(c_2_scale_global_scale, c_2_scale_raw))

                with numpyro.plate(site.n_features[0], n_features[0]):
                    """ Priors """
                    a = numpyro.sample(
                        "a", dist.TruncatedNormal(a_mean, a_scale, low=a_low)
                    )

                    b_raw = numpyro.sample("b_raw", dist.HalfNormal(scale=1))
                    b = numpyro.deterministic(site.b, jnp.multiply(b_scale, b_raw))

                    L_raw = numpyro.sample("L_raw", dist.HalfNormal(scale=1))
                    L = numpyro.deterministic(site.L, jnp.multiply(L_scale, L_raw))

                    # ell_raw = numpyro.sample("ell_raw", dist.HalfNormal(scale=1))
                    # ell = numpyro.deterministic("ell", jnp.multiply(ell_scale, ell_raw))
                    ell = numpyro.deterministic(site.ell, jnp.tile(ell_baseline, (n_features[1], n_features[1], 1)))

                    H_raw = numpyro.sample("H_raw", dist.HalfNormal(scale=1))
                    H = numpyro.deterministic(site.H, jnp.multiply(H_scale, H_raw))

                    c_1_raw = numpyro.sample("c_1_raw", dist.HalfNormal(scale=1))
                    c_1 = numpyro.deterministic(site.c_1, jnp.multiply(c_1_scale, c_1_raw))

                    c_2_raw = numpyro.sample("c_2_raw", dist.HalfNormal(scale=1))
                    c_2 = numpyro.deterministic(site.c_2, jnp.multiply(c_2_scale, c_2_raw))

        with numpyro.plate(site.n_response, self.n_response):
            with numpyro.plate(site.n_data, n_data):
                """ Model """
                mu = numpyro.deterministic(
                    site.mu,
                    F.rectified_logistic(
                        x=intensity,
                        a=a[feature0, feature1],
                        b=b[feature0, feature1],
                        L=L[feature0, feature1],
                        ell=ell[feature0, feature1],
                        H=H[feature0, feature1]
                    )
                )
                beta = numpyro.deterministic(
                    site.beta,
                    1/c_1[feature0, feature1] + jnp.true_divide(1/c_2[feature0, feature1], mu)
                )

                gradient = numpyro.deterministic(
                    "gradient",
                    F.prime(F.rectified_logistic,
                    intensity,
                    a[feature0, feature1],
                    b[feature0, feature1],
                    L[feature0, feature1],
                    ell[feature0, feature1],
                    H[feature0, feature1]
                    )
                )

                # Use the gradient as a deterministic value in your model
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


if __name__ == "__main__":
    # %%
    PLATFORM = "cpu"
    jax.config.update("jax_platforms", PLATFORM)
    numpyro.set_platform(PLATFORM)

    cpu_count = multiprocessing.cpu_count() - 2
    numpyro.set_host_device_count(cpu_count)
    numpyro.enable_x64()
    numpyro.enable_validation()

    # %%
    str_date = datetime.today().strftime('%Y-%m-%dT%H%M')
    str_date = '2024-09-21T2039'  # str_date = '2023-12-04T2113'
    # stim_type = 'TMS'
    stim_type = 'TSCS'
    toml_path = Path(r"/home/mcintosh/Local/gitprojects/hbmep-paper/configs/paper/tms/config.toml")
    build_dir = Path(r'/home/mcintosh/Cloud/Research/reports/2024/2024-09-20_paired_recruitment') / str_date / (
                stim_type + '_paired') / 'info'
    fig_format = 'png'
    fig_dpi = 300
    fig_size = (8, 8)

    # %%
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    logger = logging.getLogger(__name__)
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        format=FORMAT,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(Path(build_dir) / "logs.log", mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )

    # In[11]:
    if stim_type == 'TMS':
        stim_type_alt = 'TMS'
        mapping = {'RE2': 'Sub-tSCS', 'RE3': 'Supra-tSCS', 'REC': 'TMS-only', 'RE4': 'XSupra-tSCS'}
        xlim = [0, 80]
    elif stim_type == 'TSCS':
        stim_type_alt = 'TSS'
        mapping = {'RE2': 'Sub-TMS', 'RE3': 'Supra-TMS', 'REC': 'tSCS-only', 'RE4': 'XSupra-TMS'}
        xlim = [0, 80]
    else:
        raise Exception
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = build_dir
    # config.RESPONSE = ["AUC_APB", "AUC_ADM"]
    config.MCMC_PARAMS["num_warmup"] = 5000
    config.MCMC_PARAMS["num_samples"] = 2000
    config.FEATURES = ["protocol", "participant"]
    config.INTENSITY = stim_type + 'Int'

    model = LearnPosterior(config=config)

    # In[12]:
    ni = False
    if ni:
        src = "/home/mcintosh/Cloud/Research/reports/2024/2023-11-30_paired_recruitment/data/data_ni_2023-11_29/proc_2023-11-29_paired.csv"
        df = pd.read_csv(src)

        src = "/home/mcintosh/Cloud/Research/reports/2024/2023-11-30_paired_recruitment/data/data_ni_2023-11_29/proc_2023-11-29_paired.npy"
        mat = np.load(src)

        subset = ["SCA04", "SCA07", "SCA11"]
        #
        ind = df[model.features[1]].isin(subset)
        df = df[ind].reset_index(drop=True).copy()
        mat = mat[ind, ...]

        ind = df['stim_type'].isin([stim_type_alt])
        df = df[ind].reset_index(drop=True).copy()
        mat = mat[ind, ...]

        df, encoder_dict = model.load(df=df)

        orderby = lambda x: (x[1], x[0])
    else:

        csv_path = '/home/mcintosh/Cloud/Research/reports/2024/2024-09-20_paired_recruitment/data/data_io/np_scapptio098_2drecruit/out_info.csv'
        toml_path = '/home/mcintosh/Cloud/Research/reports/2024/2024-09-20_paired_recruitment/data/data_io/np_scapptio098_2drecruit/out_info.toml'
        df = pd.read_csv(csv_path)

        df = df[(df['datetime_posix'] < 63132884.5017998) | (df['datetime_posix'] >= 63132941.538105)]  # not sure...
        df = df.reset_index(drop=True).copy()

        # Convert 'set_group' to string
        df['set_group'] = df['set_group'].astype(str)

        with open(toml_path, 'r') as f:
            toml_data = toml.load(f)
        channels = toml_data.get('channels', [])
        rename_dict = {f'auc_sccx_{i + 1}': f'AUC_{channel}' for i, channel in enumerate(channels)}
        df.rename(columns=rename_dict, inplace=True)

        g = list(rename_dict.values())
        target_side = 'L'
        for gg in g:
            if gg.startswith(f'AUC_{target_side}'):
                gg_ = gg.replace(f'AUC_{target_side}', 'AUC_')
                df.loc[:, gg_] = df.loc[:, gg].copy()

        if stim_type == 'TMS':
            set_groups1 = ['gcxramp-001', 'gcxramp-002', 'gcxramp-003', 'gcxramp-004']
            alt_names1 = ['REC', 'RE2', 'RE3', 'RE4']
        elif stim_type == 'TSCS':
            set_groups1 = ['gscramp-001', 'gscramp-002']
            alt_names1 = ['REC', 'RE2']

        for ix, set_group in enumerate(set_groups1):
            case_local = df['set_group'] == set_group
            df.loc[case_local, 'stim_type'] = stim_type_alt
            df.loc[case_local, 'protocol'] = alt_names1[ix]

        if stim_type == 'TMS':
            # hack specifici to this dataset becaause I missed CX baselines
            vec_target_protocol = ['RE2', 'RE3', 'RE4']
            for target_protocol in vec_target_protocol:
                tolerance = 1.5e-4
                sc_int = df.loc[df.loc[:, 'protocol'] == target_protocol, 'sc_current'].mean()
                mask = (df['set_group'] == 'gscramp-001') & (np.abs(df['sc_current'] - sc_int) <= tolerance)
                df.loc[mask, 'protocol'] = target_protocol
                df.loc[mask, 'stim_type'] = stim_type_alt

        # df.rename(columns={'alias': 'participant'}, inplace=True)
        df.rename(columns={'cx_voltage': 'TMSInt'}, inplace=True)
        df.rename(columns={'sc_current': 'TSCSInt'}, inplace=True)

        df['TSCSInt'] = df['TSCSInt'] * 1e3 * 10  # mA to something on the order 0-100

        df = df[np.invert(df.loc[:, config.RESPONSE].isna().any(axis=1))]


        ind = df['stim_type'].isin([stim_type_alt])
        df = df[ind].reset_index(drop=True).copy()

        if stim_type == 'TMS':
            for target_protocol in vec_target_protocol:
                mask = (df['protocol'] == target_protocol)
                plt.plot(df.loc[mask, 'TMSInt'], df.loc[mask, 'AUC_FCR'], 'o')
                plt.show()

        df, encoder_dict = model.load(df=df)

        orderby = lambda x: (x[1], x[0])

        mat = None

    # %%
    dest = Path(build_dir).parent / "inference"
    if not os.path.exists(dest):
        os.makedirs(dest)
    p_pkl = dest / "inference.pkl"
    if os.path.isfile(p_pkl):
        logger.info('Loading model.')
        with open(p_pkl, "rb") as g:
            model, mcmc, posterior_samples = pickle.load(g)
    else:
        model.plot(df=df, encoder_dict=encoder_dict, mep_matrix=mat)

        mcmc, posterior_samples = model.run_inference(df=df)
        logger.info('Done running.')

        with open(p_pkl, "wb") as f:
            pickle.dump((model, mcmc, posterior_samples), f)

        az.to_netcdf(mcmc, str(dest / "inference.nc"))

    # In[15]:
    _posterior_samples = posterior_samples.copy()
    _posterior_samples["outlier_prob"] = _posterior_samples["outlier_prob"] * 0

    # prediction_df = model.make_prediction_dataset(df=df)
    prediction_df = model.make_prediction_dataset(df=df, min_intensity=0, max_intensity=100)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=_posterior_samples)
    # Ask Vishu if these can also return the path -
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
    prediction_df = model.make_prediction_dataset(df=df, min_intensity=xlim[0], max_intensity=xlim[1])

    df_template = prediction_df.copy()
    ind1 = df_template[model.features[1]].isin([0])
    ind2 = df_template[model.features[0]].isin([0])  # the 0 index on list is because it is a list
    df_template = df_template[ind1 & ind2]

    n_muscles = len(model.response)
    conditions = list(encoder_dict[model.features[0]].inverse_transform(np.unique(df[model.features])))
    assert conditions[-1] == "REC", 'LAST ONE HAS TO BE REC!!!'
    conditions = [mapping[conditions[ix]] for ix in range(len(conditions))]
    participants = list(encoder_dict[model.features[1]].inverse_transform(np.unique(df[model.features[1]])))

    colors = sns.color_palette('colorblind')
    pp = [[None for _ in range(len(conditions))] for _ in range(len(participants))]
    for p in range(len(participants)):
        for f in range(len(conditions)):
            df_local = df_template.copy()
            df_local[model.features[1]] = p
            df_local[model.features[0]] = f
            pp[p][f] = model.predict(df=df_local, posterior_samples=_posterior_samples)

    # %%
    _posterior_samples["H+L"] = np.zeros(_posterior_samples['H'].shape)
    for ix_p in range(len(participants)):
        for ix_muscle in range(n_muscles):
            for ix_cond in range(len(conditions)):
                _posterior_samples["H+L"][:, ix_cond, ix_p, ix_muscle] = (
                        _posterior_samples[site.L][:, ix_cond, ix_p, ix_muscle] + _posterior_samples[site.H][:, ix_cond, ix_p, ix_muscle])

    # %%
    _posterior_samples['max_grad'] = np.zeros(_posterior_samples['H'].shape)
    for ix_p in range(len(participants)):
        for ix_muscle in range(n_muscles):
            for ix_cond in range(len(conditions)):
                Y = pp[ix_p][ix_cond]['gradient'][:, :, ix_muscle]  # not sure why this index is flipped...
                _posterior_samples['max_grad'][:, ix_cond, ix_p, ix_muscle] = np.max(Y, axis=1)

    # %%
    _posterior_samples['max_facilitation_value'] = np.zeros(_posterior_samples['H'].shape)
    _posterior_samples['max_facilitation_location'] = np.zeros(_posterior_samples['H'].shape)
    fig, axs = plt.subplots(len(participants), n_muscles, figsize=(15, 10), squeeze=False)
    for ix_p in range(len(participants)):
        for ix_muscle in range(n_muscles):
            ax = axs[ix_p, ix_muscle]
            for ix_cond in range(len(conditions) - 1):
                first_column_active = pp[ix_p][ix_cond][site.mu][:, :, ix_muscle][:, 0].reshape(-1, 1)
                first_column_base = pp[ix_p][-1][site.mu][:, :, ix_muscle][:, 0].reshape(-1, 1)
                Y = ((pp[ix_p][ix_cond][site.mu][:, :, ix_muscle] + first_column_base) /
                     (first_column_active + pp[ix_p][-1][site.mu][:, :, ix_muscle]))
                a_mea = np.mean(_posterior_samples[site.a][:, ix_cond, ix_p, ix_muscle])
                Y = (Y - 1) * 100
                x = df_template[model.intensity].values
                x = (x / a_mea) * 100
                y = np.mean(Y, 0)
                _posterior_samples['max_facilitation_value'][:, ix_cond, ix_p, ix_muscle] = np.max(Y, axis=1)
                ix_am = np.argmax(Y, axis=1)
                x_ = x[ix_am]
                x_[ix_am == 0] = np.nan  # ignore cases where you get instant suppression and you don't recover... not sure about this
                _posterior_samples['max_facilitation_location'][:, ix_cond, ix_p, ix_muscle] = x_

    # %% Basic recruitment curve
    fig, axs = plt.subplots(len(participants), n_muscles, figsize=fig_size, squeeze=False)
    for ix_p in range(len(participants)):
        for ix_muscle in range(n_muscles):
            ax = axs[ix_p, ix_muscle]
            for ix_cond in range(len(conditions)):
                x = df_template[model.intensity].values
                Y = pp[ix_p][ix_cond][site.mu][:, :, ix_muscle]
                y = np.mean(Y, 0)
                y1 = np.percentile(Y, 2.5, axis=0)
                y2 = np.percentile(Y, 97.5, axis=0)
                ax.plot(x, y, color=colors[ix_cond], label=conditions[ix_cond])
                ax.fill_between(x, y1, y2, color=colors[ix_cond], alpha=0.3)

                df_local = df.copy()
                ind1 = df_local[model.features[1]].isin([ix_p])
                ind2 = df_local[model.features[0]].isin([ix_cond])  # the 0 index on list is because it is a list
                df_local = df_local[ind1 & ind2]
                x = df_local[model.intensity].values
                y = df_local[model.response[ix_muscle]].values
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
                    ax.set_xlabel(model.intensity + ' Intensity (%/mA)')
                ax.set_xlim(xlim)


    plt.show()
    # fig.savefig(Path(model.build_dir) / "REC.svg", format='svg')
    fig.savefig(Path(model.build_dir) / f"REC.{fig_format}", format=fig_format, dpi=fig_dpi)

    # %% Recruitment curve - abscissa rescaled to MT.
    fig, axs = plt.subplots(len(participants), n_muscles, figsize=fig_size, squeeze=False)
    for ix_p in range(len(participants)):
        for ix_muscle in range(n_muscles):
            ax = axs[ix_p, ix_muscle]
            for ix_cond in range(len(conditions)):
                Y = pp[ix_p][ix_cond][site.mu][:, :, ix_muscle]
                a_mea = np.mean(_posterior_samples[site.a][:, ix_cond, ix_p, ix_muscle])
                x = df_template[model.intensity].values
                x = (x/a_mea) * 100
                y = np.mean(Y, 0)
                y1 = np.percentile(Y, 2.5, axis=0)
                y2 = np.percentile(Y, 97.5, axis=0)
                ax.plot(x, y, color=colors[ix_cond], label=conditions[ix_cond])
                ax.fill_between(x, y1, y2, color=colors[ix_cond], alpha=0.3)

                df_local = df.copy()
                ind1 = df_local[model.features[1]].isin([ix_p])
                ind2 = df_local[model.features[0]].isin([ix_cond])  # the 0 index on list is because it is a list
                df_local = df_local[ind1 & ind2]
                x = df_local[model.intensity].values
                x = (x/a_mea) * 100
                y = df_local[model.response[ix_muscle]].values
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
                    ax.set_xlabel(model.intensity + ' Intensity (% MT)')
                ax.set_xlim([50, 200])

    plt.show()
    # fig.savefig(Path(model.build_dir) / "REC_MT.svg", format='svg')
    fig.savefig(Path(model.build_dir) / f"REC_MT.{fig_format}", format=fig_format, dpi=fig_dpi)

    # %%
    fig, axs = plt.subplots(len(participants), n_muscles, figsize=fig_size, squeeze=False)
    for ix_p in range(len(participants)):
        for ix_muscle in range(n_muscles):
            ax = axs[ix_p, ix_muscle]
            for ix_cond in range(len(conditions)):
                x = df_template[model.intensity].values
                Y = pp[ix_p][ix_cond]['gradient'][:, :, ix_muscle]  # not sure why this index is flipped...
                y = np.mean(Y, 0)
                y1 = np.percentile(Y, 2.5, axis=0)
                y2 = np.percentile(Y, 97.5, axis=0)
                ax.plot(x, y, color=colors[ix_cond], label=conditions[ix_cond])
                ax.fill_between(x, y1, y2, color=colors[ix_cond], alpha=0.3)

                if ix_p == 0 and ix_muscle == 0:
                    ax.legend()
                if ix_p == 0:
                    ax.set_title(config.RESPONSE[ix_muscle].split('_')[1])
                if ix_muscle == 0:
                    ax.set_ylabel('AUC (uVs)')
                    ax.set_xlabel(model.intensity + ' Intensity (%)')
                ax.set_xlim(xlim)

    plt.show()
    # fig.savefig(Path(model.build_dir) / "REC_GRAD.svg", format='svg')
    fig.savefig(Path(model.build_dir) / f"REC_GRAD.{fig_format}", format=fig_format, dpi=fig_dpi)

    # %%
    list_params = [site.a, site.H, 'max_grad', 'H+L']
    for ix_params in range(len(list_params)):
        str_p = list_params[ix_params]
        fig, axs = plt.subplots(len(participants), n_muscles, figsize=fig_size, squeeze=False)
        for ix_p in range(len(participants)):
            for ix_muscle in range(n_muscles):
                for ix_cond in range(len(conditions)):
                    ax = axs[ix_p, ix_muscle]
                    x = df_template[model.intensity].values
                    Y = _posterior_samples[str_p][:, ix_cond, ix_p, ix_muscle]
                    case_isfinite = np.isfinite(Y)
                    if len(case_isfinite) - np.sum(case_isfinite) != 0:
                        Y = Y[case_isfinite]
                        logger.info(f'{str_p} - number of non-finite vals. = {len(case_isfinite) - np.sum(case_isfinite)}!')
                    if np.all(Y == 0):
                        x_grid = np.linspace(0, 100, 1000)
                        density = np.zeros(np.shape(x_grid))
                    else:
                        kde = stats.gaussian_kde(Y)
                        x_grid = np.linspace(min(Y) * 0.9, max(Y) * 1.1, 1000)
                        density = kde(x_grid)

                    ax.plot(x_grid, density, color=colors[ix_cond], label=conditions[ix_cond])
                    # sns.histplot(Y, ax=ax)
                    if ix_p == 0 and ix_muscle == 0:
                        ax.legend()
                    if ix_p == 0:
                        ax.set_title(config.RESPONSE[ix_muscle].split('_')[1])
                    if ix_muscle == 0:
                        ax.set_xlabel(str_p)
                    # ax.set_xlim([0, 200])
                    ax.grid(which='major', color=np.ones((1, 3)) * 0.5, linestyle='--')
                    ax.grid(which='minor', color=np.ones((1, 3)) * 0.5, linestyle='--')

        plt.show()
        # fig.savefig(Path(model.build_dir) / f"param_{str_p}.svg", format='svg')
        fig.savefig(Path(model.build_dir) / f"param_{str_p}.{fig_format}", format=fig_format, dpi=fig_dpi)
        plt.close()

# %%
    list_params = ['max_facilitation_value', 'max_facilitation_location']
    for ix_params in range(len(list_params)):
        str_p = list_params[ix_params]
        fig, axs = plt.subplots(len(participants), n_muscles, figsize=fig_size, squeeze=False)
        for ix_p in range(len(participants)):
            for ix_muscle in range(n_muscles):
                for ix_cond in range(0, 1):
                    ax = axs[ix_p, ix_muscle]
                    x = df_template[model.intensity].values
                    Y = _posterior_samples[str_p][:, ix_cond, ix_p, ix_muscle]
                    case_isfinite = np.isfinite(Y)
                    if len(case_isfinite) - np.sum(case_isfinite) != 0:
                        Y = Y[case_isfinite]
                        logger.info(f'{str_p} - number of non-finite vals. = {len(case_isfinite) - np.sum(case_isfinite)}!')
                    if np.all(Y == 0):
                        x_grid = np.linspace(0, 100, 1000)
                        density = np.zeros(np.shape(x_grid))
                        x02p5 = 0
                        x97p5 = 0
                    else:
                        kde = stats.gaussian_kde(Y)
                        x_grid = np.linspace(0, 200, 1000)
                        density = kde(x_grid)
                        x02p5 = x_grid[np.argmin(abs(np.cumsum(density) / np.sum(density) - 2.5e-2))]
                        x97p5 = x_grid[np.argmin(abs(np.cumsum(density) / np.sum(density) - 97.5e-2))]

                    ax.plot(x_grid, density, color=colors[ix_cond], label=conditions[ix_cond])
                    # sns.histplot(Y, ax=ax)
                    yl = ax.get_ylim()
                    ax.plot(x02p5 * np.ones(2), yl, color='black', linestyle='--')
                    ax.plot(x97p5 * np.ones(2), yl, color='black', linestyle='--')
                    ax.text(x97p5, yl[1] * 0.75, f'{x02p5:0.0f}-{x97p5:0.0f}%', color='red')
                    if ix_p == 0 and ix_muscle == 0:
                        ax.legend()
                    if ix_p == 0:
                        ax.set_title(config.RESPONSE[ix_muscle].split('_')[1])
                    if ix_muscle == 0:
                        ax.set_xlabel(str_p)
                    ax.set_xlim([0, 200])
                    ax.set_ylim(yl)
                    ax.grid(which='major', color=np.ones((1, 3)) * 0.5, linestyle='--')

    plt.show()
    # fig.savefig(Path(model.build_dir) / f"param_{str_p}.svg", format='svg')
    fig.savefig(Path(model.build_dir) / f"param_{str_p}.{fig_format}", format=fig_format, dpi=fig_dpi)
    plt.close()

    # %%
    fig, axs = plt.subplots(len(participants), n_muscles, figsize=fig_size, squeeze=False)
    for ix_p in range(len(participants)):
        for ix_muscle in range(n_muscles):
            ax = axs[ix_p, ix_muscle]
            for ix_cond in range(len(conditions) - 1):
                first_column_active = pp[ix_p][ix_cond][site.mu][:, :, ix_muscle][:, 0].reshape(-1, 1)
                first_column_base = pp[ix_p][-1][site.mu][:, :, ix_muscle][:, 0].reshape(-1, 1)
                Y = ((pp[ix_p][ix_cond][site.mu][:, :, ix_muscle] + first_column_base) /
                     (first_column_active + pp[ix_p][-1][site.mu][:, :, ix_muscle]))

                Y = (Y - 1) * 100
                x = df_template[model.intensity].values
                y = np.mean(Y, 0)
                y1 = np.percentile(Y, 2.5, axis=0)
                y2 = np.percentile(Y, 97.5, axis=0)
                a_mea = np.mean(_posterior_samples[site.a][:, ix_cond, ix_p, ix_muscle])
                x = (x / a_mea) * 100
                ax.plot(x, y, color=colors[ix_cond], label=conditions[ix_cond])
                ax.fill_between(x, y1, y2, color=colors[ix_cond], alpha=0.3)

                if ix_p == 0 and ix_muscle == 0:
                    ax.legend()
                    ax.set_ylabel('% Fac. (Paired/Brain-only + Spine-only)')
                if ix_p == 0:
                    ax.set_title(model.response[ix_muscle].split('_')[1])
                if ix_muscle == 0:
                    ax.set_xlabel('Threshold %')
                ax.set_xlim([50, 200])
                ax.set_axisbelow(True)
                ax.grid(which='major', color=np.ones((1, 3)) * 0.5, linestyle='--')

                if ix_cond == 0:
                    x_max = x[np.argmax(y)]
                    ax.plot(x_max * np.ones(2), ax.get_ylim(), color=colors[ix_cond], linestyle='--')
                    y_text = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.75
                    ax.text(x_max, y_text, f"{x_max:0.1f}%")
                ax.set_xlim([50, 200])

    plt.show()
    # fig.savefig(Path(model.build_dir) / "REC_MT_cond_norm.svg", format='svg')
    fig.savefig(Path(model.build_dir) / f"REC_MT_cond_norm.{fig_format}", format=fig_format, dpi=fig_dpi)

    # %%
    fig, axs = plt.subplots(len(participants), n_muscles, figsize=fig_size, squeeze=False)
    for ix_p in range(len(participants)):
        for ix_muscle in range(n_muscles):
            ax = axs[ix_p, ix_muscle]
            for ix_cond in range(len(conditions) - 1):
                first_column_active = pp[ix_p][ix_cond][site.mu][:, :, ix_muscle][:, 0].reshape(-1, 1)
                first_column_base = pp[ix_p][-1][site.mu][:, :, ix_muscle][:, 0].reshape(-1, 1)
                Y = ((pp[ix_p][ix_cond][site.mu][:, :, ix_muscle] + first_column_base) -
                     (first_column_active + pp[ix_p][-1][site.mu][:, :, ix_muscle]))

                x = df_template[model.intensity].values
                y = np.mean(Y, 0)
                y1 = np.percentile(Y, 2.5, axis=0)
                y2 = np.percentile(Y, 97.5, axis=0)
                a_mea = np.mean(_posterior_samples[site.a][:, ix_cond, ix_p, ix_muscle])
                x = (x / a_mea) * 100
                ax.plot(x, y, color=colors[ix_cond], label=conditions[ix_cond])
                ax.fill_between(x, y1, y2, color=colors[ix_cond], alpha=0.3)

                if ix_p == 0 and ix_muscle == 0:
                    ax.legend()
                    ax.set_ylabel('AUC: Paired - (Brain-only + Spine-only)')
                if ix_p == 0:
                    ax.set_title(model.response[ix_muscle].split('_')[1])
                if ix_muscle == 0:
                    ax.set_xlabel('Threshold %')
                ax.set_xlim([50, 200])
                ax.set_axisbelow(True)
                ax.grid(which='major', color=np.ones((1, 3)) * 0.5, linestyle='--')

                if ix_cond == 0:
                    x_max = x[np.argmax(y)]
                    ax.plot(x_max * np.ones(2), ax.get_ylim(), color=colors[ix_cond], linestyle='--')
                    y_text = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.75
                    ax.text(x_max, y_text, f"{x_max:0.1f}")
                ax.set_xlim([50, 200])

    plt.show()
    # fig.savefig(Path(model.build_dir) / "REC_MT_cond_sub.svg", format='svg')
    fig.savefig(Path(model.build_dir) / f"REC_MT_cond_sub.{fig_format}", format=fig_format, dpi=fig_dpi)

    # %%
    command = f"find {build_dir.parent} -name '*.pdf' -exec sh -c '[ ! -f \"${{0%.pdf}}.png\" ] && convert \"$0\" \"${{0%.pdf}}.png\"' {{}} \;" # command = f"find {build_dir.parent} -name '*.pdf' -exec sh -c 'convert \"$0\" \"${{0%.pdf}}.png\"' {{}} \;"
    subprocess.run(command, shell=True, check=True)

    # In[13]:
    numpyro_data = az.from_numpyro(mcmc)
    score = az.loo(numpyro_data)
    str_out = f"ELPD LOO (Log): {score.elpd_loo:.2f} for {str_date}"
    print(str_out)
    logger.info(str_out)

