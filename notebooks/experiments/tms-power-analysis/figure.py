import os
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep_paper.utils import setup_logging
from models import (
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
)
from core import N_SUBJECTS_SPACE
from constants import (
    EXPERIMENTS_DIR,
    EXPERIMENTS_NO_EFFECT_DIR
)

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"

BUILD_DIR = EXPERIMENTS_DIR
BAYESIAN_CUTOFF = .95
FREQUENTIST_CUTOFF = .05


def main():
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.566, 2.65),
        squeeze=False,
        constrained_layout=True
    )

    n_subjects_space = N_SUBJECTS_SPACE[1:]
    logger.info(f"n_subjects_space: {n_subjects_space}")
    models = [
        NonHierarchicalBayesianModel,
        HierarchicalBayesianModel
    ]
    labels = [
        "Non-Hierarchical Bayesian",
        "Hierarchical Bayesian"
    ]

    src = os.path.join(EXPERIMENTS_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(f"mae: {mae.shape}")

    src = os.path.join(EXPERIMENTS_DIR, "prob.npy")
    prob = np.load(src)
    logger.info(f"prob: {prob.shape}")

    ax = axes[0, 0]
    for model_ind, model in enumerate(models):
        x = n_subjects_space
        y = (
            prob[..., model_ind] > BAYESIAN_CUTOFF
            if model_ind == 1
            else prob[..., model_ind] < FREQUENTIST_CUTOFF
        )
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ax.errorbar(
            x=[model_ind / 3 + i for i in x],
            y=yme,
            yerr=ysem,
            marker="o",
            label=f"{model.NAME}",
            linestyle="--",
            # ms=1,
            # linewidth=1,
            # color=colors[model_ind]
        )
    ax.axhline(y=.80, linestyle="--", color="black")
    ax.set_xticks(x)
    ax.legend(bbox_to_anchor=(0., 1.2), loc="center", fontsize=6)
    ax.set_yticks([.20 * i for i in range(6)])
    ax.set_xlabel("# Subjects")
    ax.set_ylabel("Power (β)")

    src = os.path.join(EXPERIMENTS_NO_EFFECT_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(f"mae: {mae.shape}")

    src = os.path.join(EXPERIMENTS_NO_EFFECT_DIR, "prob.npy")
    prob = np.load(src)
    logger.info(f"prob: {prob.shape}")

    ax = axes[0, 1]
    for model_ind, model in enumerate(models):
        x = n_subjects_space
        y = (
            prob[..., model_ind] > BAYESIAN_CUTOFF
            if model_ind == 1
            else prob[..., model_ind] < FREQUENTIST_CUTOFF
        )
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ax.errorbar(
            x=[model_ind / 3 + i for i in x],
            y=yme,
            yerr=2 * ysem,
            marker="o",
            label=f"{model.NAME}",
            linestyle="--",
            # ms=1,
            # linewidth=1,
            # color=colors[model_ind]
        )
    ax.axhline(y=.05, linestyle="--", color="black")
    ax.set_xticks(x)
    ax.legend(bbox_to_anchor=(0., 1.2), loc="center", fontsize=6)
    ax.set_yticks([.02 * i for i in range(6)])
    ax.set_xlabel("# Subjects")
    ax.set_ylabel("Power (β)")

    dest = os.path.join(EXPERIMENTS_DIR, "results.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    # ax = axes[0, 0]
    # for model_ind, model in enumerate(models):
    #     # if model_ind == 3: continue
    #     x = n_subjects_space
    #     # Jitter x
    #     # x = [model_ind / 100 + i for i in x]
    #     y = mae[..., model_ind]
    #     yme = y.mean(axis=-1)
    #     ysem = stats.sem(y, axis=-1)
    #     # ysd = y.std(axis=-1)
    #     ax.errorbar(
    #         x=x,
    #         y=yme,
    #         yerr=ysem,
    #         marker="o",
    #         label=labels[model_ind],
    #         linestyle="--",
    #         ms=4,
    #         # linewidth=1,
    #         color=colors[model_ind]
    #     )
    #     ax.set_xticks(x)
    #     ax.set_xlabel("# Subjects")
    #     ax.set_ylabel("MAE")

    # ax.set_ylim(bottom=0.)
    # ax.set_xlabel("Number of Participants", fontsize=axis_label_size)
    # ax.set_ylabel("MAE on Threshold $($% MSO$)$", fontsize=axis_label_size)

    # n_pulses_space = N_PULSES_SPACE
    # src = os.path.join(NUMBER_OF_PULSES_DIR, "mae.npy")
    # mae = np.load(src)

    # ax = axes[0, 1]
    # for model_ind, model in enumerate(models):
    #     # if model_ind == 3: continue
    #     x = n_pulses_space
    #     # Jitter x
    #     # x = [model_ind / 100 + i for i in x]
    #     y = mae[..., model_ind]
    #     yme = y.mean(axis=-1)
    #     ysem = stats.sem(y, axis=-1)
    #     # ysd = y.std(axis=-1)
    #     ax.errorbar(
    #         x=x,
    #         y=yme,
    #         yerr=ysem,
    #         marker="o",
    #         label=labels[model_ind],
    #         linestyle="--",
    #         ms=4,
    #         # linewidth=1,
    #         color=colors[model_ind]
    #     )
    #     ax.set_xticks(x)
    #     # ax.legend(bbox_to_anchor=(0., 1.2), loc="center", fontsize=6)
    #     ax.set_xlabel("# Pulses")
    #     ax.set_ylabel("MAE")

    # ax.set_ylim(bottom=0.)
    # ax.set_xlabel("Number of Intensities", fontsize=axis_label_size)
    # ax.set_ylabel("MAE on Threshold $($% MSO$)$", fontsize=axis_label_size)

    # for i in range(ncols):
    #     ax = axes[0, i]
    #     sides = ["top", "right"]
    #     for side in sides:
    #         ax.spines[side].set_visible(False)
    #     ax.tick_params(
    #         axis='both',
    #         which='both',
    #         left=True,
    #         bottom=True,
    #         right=False,
    #         top=False,
    #         labelleft=True,
    #         labelbottom=True,
    #         labelright=False,
    #         labeltop=False,
    #         labelrotation=15,
    #         labelsize=8
    #     )
    #     ax.grid(axis="y", linestyle="--", alpha=.25)
    #     ax.set_ylabel("")

    # ax = axes[0, 0]
    # ax.sharey(axes[0, 1])
    # ax.set_ylabel("Mean Absolute Error on Threshold\n$($% MSO$)$", fontsize=axis_label_size)
    # ax.legend(loc="upper right", fontsize=6)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.get_legend().remove()
    # ax.legend(handles[:3], labels[:3], loc="upper right", fontsize=6, frameon=True)

    # ax = axes[0, 1]
    # ax.tick_params(labelleft=False)
    # ax.legend(handles[3:], labels[3:], loc="upper right", fontsize=6, frameon=True)
    # ax.set_ylim(top=10.5)

    fig.align_xlabels()
    fig.align_ylabels()

    dest = os.path.join(BUILD_DIR, "results.svg")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(BUILD_DIR, "results.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
