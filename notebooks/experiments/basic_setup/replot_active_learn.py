import functools
import os
import pickle
import logging

import pandas as pd
import numpy as np
import jax

from hbmep.config import Config
from hbmep.model.utils import Site as site

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erf

from learn_posterior import TOML_PATH
from models import RectifiedLogistic
from utils import run_inference

from scipy.stats import gaussian_kde
from scipy.integrate import nquad
from joblib import Parallel, delayed
from pathlib import Path

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def main():
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    root_dir = Path(config.BUILD_DIR)
    root_dir = Path('/media/hdd2/mcintosh/OneDrive/Other/Desktop/test12')  # TEMP

    config.MCMC_PARAMS['num_chains'] = 1
    config.MCMC_PARAMS['num_warmup'] = 500
    config.MCMC_PARAMS['num_samples'] = 1000
    seed = dict()
    seed['ix_gen_seed'] = 10
    seed['ix_participant'] = 62
    opt_param = ['a', 'H']  # ['a', 'H']
    N_max = 30
    N_obs = 15  # this is how many enropy calcs to do per every y drawn from x... larger is better
    assert N_obs % 2 != 0, "Better if N_obs is odd."

    fig_format = 'png'
    fig_dpi = 300
    fig_size = (6, 3)

    config.BUILD_DIR = root_dir / 'learn_posterior_rt2'
    simulator = RectifiedLogistic(config=config)
    # simulator._make_dir(simulator.build_dir)

    """ Load learnt posterior """
    src = config.BUILD_DIR / "inference.pkl"
    with open(src, "rb") as g:
        model, mcmc, posterior_samples, df = pickle.load(g)
    # src = config.BUILD_DIR / "entropy.pkl"
    # with open(src, "rb") as g:
    #     mat_entropy, entropy_base, next_intensity, vec_candidate_int, N_obs = pickle.load(g)
    participants = ['MEH']
    xlim = [0, 100]
    # Simulate TOTAL_SUBJECTS subjects
    TOTAL_PULSES = 100
    TOTAL_SUBJECTS = len(participants)
    conditions = ['MEH']

    # Create template dataframe for simulation
    df_custom = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[simulator.features[0]]) \
        .merge(
            pd.DataFrame([0, 90], columns=[simulator.intensity]),
            how="cross"
        )
    df_custom = simulator.make_prediction_dataset(
        df=df_custom, min_intensity=0, max_intensity=100, num=TOTAL_PULSES)

    colors = sns.color_palette('colorblind')
    pp = model.predict(df=df_custom, posterior_samples=posterior_samples)

    # df_template = prediction_df.copy()
    # ind1 = df_template[model.features[1]].isin([0])
    # ind2 = df_template[model.features[0]].isin([0])  # the 0 index on list is because it is a list
    # df_template = df_template[ind1 & ind2]

    n_muscles = len(model.response)
    # conditions = list(encoder_dict[model.features[0]].inverse_transform(np.unique(df[model.features])))
    # conditions = [mapping[conditions[ix]] for ix in range(len(conditions))]
    # participants = list(encoder_dict[model.features[1]].inverse_transform(np.unique(df[model.features[1]])))

    fig, axs = plt.subplots(1, n_muscles, figsize=fig_size)
    for ix_muscle in range(n_muscles):
        ax = axs[ix_muscle]
        x = df_custom[model.intensity].values
        Y = pp[site.mu][:, :, ix_muscle]
        y = np.mean(Y, 0)
        y1 = np.percentile(Y, 2.5, axis=0)
        y2 = np.percentile(Y, 97.5, axis=0)
        ax.plot(x, y, color=colors[ix_muscle], label=ix_muscle)
        ax.fill_between(x, y1, y2, color=colors[ix_muscle], alpha=0.3)

        df_local = df.copy()
        x = df_local[model.intensity].values
        y = df_local[model.response[ix_muscle]].values
        ax.plot(x, y,
                color=colors[ix_muscle], marker='o', markeredgecolor='w',
                markerfacecolor=colors[ix_muscle], linestyle='None',
                markeredgewidth=1, markersize=4)

        # if ix_p == 0 and ix_muscle == 0:
        #     ax.legend()
        # if ix_p == 0:
        #     ax.set_title(config.RESPONSE[ix_muscle].split('_')[1])
        if ix_muscle == 0:
            ax.set_ylabel('AUC (uVs)')
            ax.set_xlabel(model.intensity + ' Intensity (%/mA)')
        ax.set_xlim(xlim)

    fig.savefig(Path(model.build_dir) / f"REC_MT_cond_norm.{fig_format}", format=fig_format, dpi=fig_dpi)
    # plt.show()
    # fig.savefig(Path(model.build_dir) / "REC_MT_cond_norm.svg", format='svg')
    plt.close()

    list_params = [site.a, site.H]
    for ix_params in range(len(list_params)):
        str_p = list_params[ix_params]
        fig, axs = plt.subplots(1, n_muscles, figsize=fig_size)
        for ix_muscle in range(n_muscles):
            ax = axs[ix_muscle]
            x = df[model.intensity].values
            Y = posterior_samples[str_p][:, 0, ix_muscle]
            case_isfinite = np.isfinite(Y)
            if len(case_isfinite) - np.sum(case_isfinite) != 0:
                Y = Y[case_isfinite]
            if np.all(Y == 0):
                x_grid = np.linspace(0, 100, 1000)
                density = np.zeros(np.shape(x_grid))
            else:
                kde = gaussian_kde(Y)
                x_grid = np.linspace(min(Y) * 0.9, max(Y) * 1.1, 1000)
                density = kde(x_grid)

            ax.plot(x_grid, density, color=colors[ix_muscle], label=ix_muscle)
            # sns.histplot(Y, ax=ax)
            # if ix_p == 0 and ix_muscle == 0:
            #     ax.legend()
            # if ix_p == 0:
            #     ax.set_title(config.RESPONSE[ix_muscle].split('_')[1])
            if ix_muscle == 0:
                ax.set_xlabel(str_p)
            # ax.set_xlim([0, 200])
            ax.grid(which='major', color=np.ones((1, 3)) * 0.5, linestyle='--')
            ax.grid(which='minor', color=np.ones((1, 3)) * 0.5, linestyle='--')

        # plt.show()
        # fig.savefig(Path(model.build_dir) / f"param_{str_p}.svg", format='svg')
        fig.savefig(Path(model.build_dir) / f"param_{str_p}.{fig_format}", format=fig_format, dpi=fig_dpi)
        plt.close()

if __name__ == "__main__":
    main()
