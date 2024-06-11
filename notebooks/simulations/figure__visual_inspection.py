import os
import pickle
import logging

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.patches import Rectangle
import seaborn as sns

from jax import random
import pandas as pd
import numpy as np
import scipy.stats as stats

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models__accuracy import HierarchicalBayesianModel
from constants__accuracy import (
    DATA_PATH,
    LEARN_POSTERIOR_DIR,
    SIMULATE_DATA_DIR__ACCURACY,
    REP,
    INFERENCE_FILE,
    SIMULATION_DF
)
logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"

BUILD_DIR = SIMULATE_DATA_DIR__ACCURACY

SIMULATION_DF_PATH = os.path.join(SIMULATE_DATA_DIR__ACCURACY, SIMULATION_DF)
SIMULATION_PPD_PATH = os.path.join(SIMULATE_DATA_DIR__ACCURACY, INFERENCE_FILE)

N_REPS = 1

max_color, max_alpha = 255, 100
scatter_color = (179 / max_color, 179 / max_color, 179 / max_color, 100 / max_alpha)
prior_color = (.7, .7, .7)
prior_color = "#92A8D1"
observed_color = "#F7CAC9"
ppd_color = "#88B04B"

markersize = .8
axis_label_size = 12
inside_text_size = 8

@timing
def main():
    # Load learnt posterior
    src = os.path.join(LEARN_POSTERIOR_DIR, INFERENCE_FILE)
    with open(src, "rb") as f:
        model, _, posterior_samples = pickle.load(f)

    # Load real data
    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)

    combinations = [(6,), (12,)]
    ind = df[model.features].apply(tuple, axis=1).isin(combinations)
    df = df[ind].reset_index(drop=True).copy()

    # Predict
    posterior_predictive = model.predict(df=df, posterior_samples=posterior_samples)

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    # Load simulated dataframe
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    # # Load parameters reduced by PCA
    # src = os.path.join(BUILD_DIR, "pca.pkl")
    # with open(src, "rb") as f:
    #     pca, params_embedded, ppd_params_embedded, prior_params_embedded = pickle.load(f)

    # # Subsample
    # def subsample(random_key, arr, N):
    #     result = None
    #     keys = random.split(random_key, arr.shape[1])
    #     for i, key in enumerate(keys):
    #         samples = random.choice(key, arr[:, i:i + 1, ...], shape=(N,), replace=False)
    #         if result is None: result = samples
    #         else: result = np.concatenate([result, samples], axis=1)
    #     return result

    # N = 200
    # random_key = random.PRNGKey(0)
    # random_keys = random.split(random_key, 3)
    # # params_embedded = subsample(random_keys[0], params_embedded, 400)
    # # ppd_params_embedded = subsample(random_keys[1], ppd_params_embedded, 300)
    # # prior_params_embedded = subsample(random_keys[2], prior_params_embedded, 2000)
    # params_embedded = subsample(random_keys[0], params_embedded, N)
    # ppd_params_embedded = subsample(random_keys[1], ppd_params_embedded, N)
    # prior_params_embedded = subsample(random_keys[2], prior_params_embedded, 2000)

    # params_embedded = params_embedded.reshape(-1, 2)
    # ppd_params_embedded = ppd_params_embedded.reshape(-1, 2)
    # prior_params_embedded = prior_params_embedded.reshape(-1, 2)

    # Plot
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 5),
        constrained_layout=True,
        squeeze=False
    )

    for i in range(nrows):
        match i:
            case 0:
                draw = 3100
                label = "$Participant_{1}$"

            case 1:
                draw = 1500
                label = "$Participant_{N}$"

        c = combinations[i]
        ind = df[model.features].apply(tuple, axis=1).isin([c])
        temp_df = df[ind].reset_index(drop=True).copy()

        ax = axes[i, 0]
        sns.scatterplot(
            x=temp_df[model.intensity],
            y=temp_df[model.response[0]],
            ax=ax,
            color=scatter_color
        )
        ax.text(0, 5.5, label, fontsize=inside_text_size)

        ax = axes[i, 1]
        sns.scatterplot(
            x=temp_df[model.intensity],
            y=posterior_predictive[site.obs][draw, ind, 0],
            ax=ax,
            color=scatter_color
        )
        ax.text(0, 5.5, label, fontsize=inside_text_size, fontweight="semibold")

        match i:
            case 0:
                # c = (3,)
                # draw = 2
                # c = (6,)
                # draw = 24
                # c = (25,)
                # draw = 25
                c = (9,)
                draw = 1
                c = (8,)
                draw = 5
                c = (9,)
                draw = 12
                c = (23,)
                draw = 40
                c = (28,)
                draw = 20
                c = (31,)
                draw = 36
                label = "$Participant_{N + 1}$"

            case 1:
                c = (2,)
                draw = 0
                label = "$Participant_{N + M}$"

        ax = axes[i, 2]
        ind = (
            (simulation_df[model.features].apply(tuple, axis=1).isin([c]))
            & (simulation_df[REP] < N_REPS)
        )
        temp_simulation_df = simulation_df[ind].reset_index(drop=True).copy()
        sns.scatterplot(
            x=temp_simulation_df[model.intensity],
            y=simulation_ppd[site.obs][draw, ind, 0],
            ax=ax,
            color=scatter_color
        )
        ax.axvline(
            simulation_ppd[site.a][draw, *c, 0],
            ymax=.48,
            color=ppd_color,
            label="True Threshold",
            linestyle="--"
        )
        ax.text(0, 5.5, label, fontsize=inside_text_size, fontweight="semibold")

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            ax.sharex(axes[0, 0])
            ax.sharey(axes[0, 0])

    ax = axes[0, 0]
    ax.set_xlim(right=105)
    ax.set_ylim(top=6.3)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 3, 6])

    ax = axes[0, 0]
    ax.set_title("Observed participants\n(Real data)", fontsize=axis_label_size)

    ax = axes[0, 1]
    ax.set_title("Observed participants\n(Simulated data)", fontsize=axis_label_size)

    ax = axes[0, 2]
    ax.set_title("New participants\n(Simulated data)", fontsize=axis_label_size)

    # ax = axes[0, 3]
    # sns.scatterplot(
    #     x=prior_params_embedded[:, 0],
    #     y=prior_params_embedded[:, 1],
    #     ax=ax,
    #     label="Prior parameters",
    #     color=prior_color,
    #     s=markersize
    # )
    # sns.scatterplot(
    #     x=params_embedded[:, 0],
    #     y=params_embedded[:, 1],
    #     ax=ax,
    #     label="Observed participants' parameters",
    #     color=observed_color,
    #     s=markersize,
    # )
    # sns.scatterplot(
    #     x=ppd_params_embedded[:, 0],
    #     y=ppd_params_embedded[:, 1],
    #     ax=ax,
    #     label="Simulated parameters",
    #     color=ppd_color,
    #     s=markersize,
    # )
    # w = 6
    # a, b = -5, -6.2
    # # ax.add_patch(Rectangle((a, b), 2 * w, 2 * w, edgecolor="black", facecolor="none", linewidth=1.5))
    # # ax.set_yticks([15, 0, -15])
    # # ax.set_xticks([-20, 0, 20])
    # ax.set_xlim(-5, 12)
    # ax.set_ylim(-5, 12)
    # # ax.set_xlim(-5, 5)
    # # ax.set_ylim(-5, 5)
    # # ax.set_xlim([-2.5, 2.5])
    # # ax.set_ylim([-2.5, 2.5])

    # ax = axes[1, 3]
    # sns.scatterplot(
    #     x=params_embedded[:, 0],
    #     y=params_embedded[:, 1],
    #     ax=ax,
    #     label="Observed Parameters",
    #     color=observed_color,
    #     s=markersize,
    # )
    # sns.scatterplot(
    #     x=ppd_params_embedded[:, 0],
    #     y=ppd_params_embedded[:, 1],
    #     label="Simulated Parameters",
    #     color=ppd_color,
    #     s=markersize,
    # )

    # # ax.set_xticks([-3, 0, 3])
    # # ax.set_yticks([-3, 0, 3])
    # # ax.set_xlim(-4, 6)
    # # ax.set_ylim(-5, 5)

    # for i in range(nrows):
    #     for j in range(ncols):
    #         ax = axes[i, j]

    #         if j == 3:
    #             sides = ["top", "right"]
    #             for side in sides:
    #                 ax.spines[side].set_visible(False)

    #         if ax.get_legend() is not None:
    #             ax.get_legend().remove()

    #         ax.tick_params(
    #             axis='both',
    #             which='both',
    #             left=True,
    #             bottom=True,
    #             right=False,
    #             top=False,
    #             labelleft=True,
    #             labelbottom=True,
    #             labelright=False,
    #             labeltop=False,
    #             labelrotation=15,
    #             labelsize=10
    #         )
    #         ax.set_ylabel("")
    #         ax.set_xlabel("")

    # lgnd = axes[0, 3].legend(loc=(0, .85), ncol=1, columnspacing=.5, fontsize=10, handletextpad=0)
    # lgnd.legend_handles[0]._sizes = [30]
    # lgnd.legend_handles[1]._sizes = [30]
    # lgnd.legend_handles[2]._sizes = [30]

    ax = axes[1, 0]
    ax.set_ylabel("MEP size (a.u.)", fontsize=axis_label_size)

    for j in range(1):
        ax = axes[1, j]
        ax.set_xlabel("Stimulation intensity ($\%$ MSO)", fontsize=axis_label_size)

    for i in range(2):
        ax = axes[i, 3]
        ax.set_ylabel("PC 2", fontsize=axis_label_size)

        if i == 1:
            ax.set_xlabel("PC 1", fontsize=axis_label_size)

    ax = axes[0, 2]
    ax.legend(loc=(0.02, 0.67), fontsize=10)

    fig.align_xlabels()
    fig.align_ylabels()

    dest = os.path.join(BUILD_DIR, "visual_inspection.svg")
    fig.savefig(dest, dpi=600)

    dest = os.path.join(BUILD_DIR, "visual_inspection.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved PCA plot to {dest}")


if __name__ == "__main__":
    setup_logging(BUILD_DIR, os.path.basename(__file__))
    main()
