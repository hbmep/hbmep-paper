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
    root_dir = Path('/media/hdd2/mcintosh/OneDrive/Other/Desktop/test06_15_a')  # TEMP

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
    fig_size = (20, 12)

    config.BUILD_DIR = root_dir / 'learn_posterior_rt2'
    simulator = RectifiedLogistic(config=config)
    # simulator._make_dir(simulator.build_dir)

    """ Load learnt posterior """
    src = config.BUILD_DIR / "inference.pkl"
    with open(src, "rb") as g:
        model, mcmc, posterior_samples, df = pickle.load(g)
    src = config.BUILD_DIR / "entropy.pkl"
    with open(src, "rb") as g:
        mat_entropy, entropy_base, next_intensity, vec_candidate_int, N_obs = pickle.load(g)
    participants = ['MEH']
    xlim = [0, 100]
    # Simulate TOTAL_SUBJECTS subjects
    TOTAL_PULSES = 100
    TOTAL_SUBJECTS = len(participants)
    conditions = ['MEH']

    # Create template dataframe for simulation
    df_local = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[simulator.features[0]]) \
        .merge(
            pd.DataFrame([0, 90], columns=[simulator.intensity]),
            how="cross"
        )
    df_local = simulator.make_prediction_dataset(
        df=df_local, min_intensity=0, max_intensity=100, num=TOTAL_PULSES)

    colors = sns.color_palette('colorblind')
    pp = model.predict(df=df_local, posterior_samples=posterior_samples)

    # df_template = prediction_df.copy()
    # ind1 = df_template[model.features[1]].isin([0])
    # ind2 = df_template[model.features[0]].isin([0])  # the 0 index on list is because it is a list
    # df_template = df_template[ind1 & ind2]

    n_muscles = len(model.response)
    # conditions = list(encoder_dict[model.features[0]].inverse_transform(np.unique(df[model.features])))
    # conditions = [mapping[conditions[ix]] for ix in range(len(conditions))]
    # participants = list(encoder_dict[model.features[1]].inverse_transform(np.unique(df[model.features[1]])))

    fig, axs = plt.subplots(n_muscles, figsize=fig_size)
    for ix_muscle in range(n_muscles):
        ax = axs[ix_muscle]
        x = df_local[model.intensity].values
        Y = pp[site.mu][:, :, ix_muscle]
        y = np.mean(Y, 0)
        y1 = np.percentile(Y, 2.5, axis=0)
        y2 = np.percentile(Y, 97.5, axis=0)
        ax.plot(x, y, color=colors[ix_muscle], label=ix_muscle)
        ax.fill_between(x, y1, y2, color=colors[ix_muscle], alpha=0.3)

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
    print(1)


if __name__ == "__main__":
    main()
