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
    MaximumLikelihoodModel,
    NelderMeadOptimization,
    SVIHierarchicalBayesianModel
)
from core__number_of_subjects import (
    N_REPS, N_PULSES, N_SUBJECTS_SPACE
)
from constants import (
    N_PULSES_SPACE,
    NUMBER_OF_SUJECTS_DIR,
    NUMBER_OF_PULSES_DIR,
    EXPERIMENTS_DIR
)

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"

BUILD_DIR = EXPERIMENTS_DIR
axis_label_size = 8


def main():
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.566, 2.65),
        squeeze=False,
        constrained_layout=True
    )

    n_subjects_space = N_SUBJECTS_SPACE
    models = [
        NelderMeadOptimization,
        MaximumLikelihoodModel,
        NonHierarchicalBayesianModel,
        SVIHierarchicalBayesianModel,
        HierarchicalBayesianModel
    ]
    labels = [
        "Nelder-Mead Optimization",
        "Maximum-Likelihood",
        "Non-Hierarchical Bayesian",
        "Hierarchical Bayesian (SVI)",
        "Hierarchical Bayesian (NUTS)"
    ]

    cmap = sns.color_palette("hls", 8)
    # colors = cmap(np.linspace(0, .7, 5 * len(models)))[::-1][::5]
    colors = [cmap[-1]]
    begin = 3
    colors += cmap[begin:begin + 4]
    colors = [colors[2], colors[1], colors[-1], colors[-2], colors[0]]

    src = os.path.join(NUMBER_OF_SUJECTS_DIR, "mae.npy")
    mae = np.load(src)

    ax = axes[0, 0]
    for model_ind, model in enumerate(models):
        # if model_ind == 3: continue
        x = n_subjects_space
        # Jitter x
        # x = [model_ind / 100 + i for i in x]
        y = mae[..., model_ind]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        # ysd = y.std(axis=-1)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=ysem,
            marker="o",
            label=labels[model_ind],
            linestyle="--",
            ms=4,
            # linewidth=1,
            color=colors[model_ind]
        )
        ax.set_xticks(x)
        ax.set_xlabel("# Subjects")
        ax.set_ylabel("MAE")

    ax.set_ylim(bottom=0.)
    ax.set_xlabel("Number of Participants", fontsize=axis_label_size)
    ax.set_ylabel("MAE on Threshold $($% MSO$)$", fontsize=axis_label_size)

    n_pulses_space = N_PULSES_SPACE
    src = os.path.join(NUMBER_OF_PULSES_DIR, "mae.npy")
    mae = np.load(src)

    ax = axes[0, 1]
    for model_ind, model in enumerate(models):
        # if model_ind == 3: continue
        x = n_pulses_space
        # Jitter x
        # x = [model_ind / 100 + i for i in x]
        y = mae[..., model_ind]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        # ysd = y.std(axis=-1)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=ysem,
            marker="o",
            label=labels[model_ind],
            linestyle="--",
            ms=4,
            # linewidth=1,
            color=colors[model_ind]
        )
        ax.set_xticks(x)
        # ax.legend(bbox_to_anchor=(0., 1.2), loc="center", fontsize=6)
        ax.set_xlabel("# Pulses")
        ax.set_ylabel("MAE")

    ax.set_ylim(bottom=0.)
    ax.set_xlabel("Number of Pulses", fontsize=axis_label_size)
    ax.set_ylabel("MAE on Threshold $($% MSO$)$", fontsize=axis_label_size)

    for i in range(ncols):
        ax = axes[0, i]
        sides = ["top", "right"]
        for side in sides:
            ax.spines[side].set_visible(False)
        ax.tick_params(
            axis='both',
            which='both',
            left=True,
            bottom=True,
            right=False,
            top=False,
            labelleft=True,
            labelbottom=True,
            labelright=False,
            labeltop=False,
            labelrotation=15,
            labelsize=8
        )
        ax.grid(axis="y", linestyle="--", alpha=.25)
        ax.set_ylabel("")

    ax = axes[0, 0]
    ax.sharey(axes[0, 1])
    ax.set_ylabel("Mean Absolute Error\non Threshold $($% MSO$)$", fontsize=axis_label_size)
    ax.legend(loc="upper right", fontsize=6)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.legend(handles[:3], labels[:3], loc="upper right", fontsize=7, frameon=False)

    ax = axes[0, 1]
    ax.tick_params(labelleft=False)
    ax.legend(handles[3:], labels[3:], loc="upper right", fontsize=7, frameon=False)
    ax.set_ylim(top=10.5)

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
