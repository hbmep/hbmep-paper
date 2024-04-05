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

axis_label_size = 8

cmap = sns.color_palette("hls", 8)
colors = [cmap[-1]]
begin = 3
colors += cmap[begin:begin + 4]
colors = [colors[2], colors[1], colors[-2], colors[0], "k"]
colors = [colors[-3], colors[-1]]

markersize = 3
linewidth = 1
linestyle = "--"


def _power_plot(ax, x, arr, models, labels):
    assert len(x) == arr.shape[0]
    for model_ind, model in enumerate(models):
        match model.NAME:
            case HierarchicalBayesianModel.NAME:
                # y = arr[..., model_ind] > BAYESIAN_CUTOFF
                y = arr[..., model_ind] < 0.
            case NonHierarchicalBayesianModel.NAME:
                y = arr[..., model_ind] < FREQUENTIST_CUTOFF
            case _:
                raise ValueError(f"Invalid model")

        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=ysem,
            marker="o",
            label=labels[model_ind],
            linestyle=linestyle,
            ms=markersize,
            linewidth=linewidth,
            color=colors[model_ind],
        )

    ax.set_xticks(x)
    return ax


def _error_plot(ax, x, arr, models, labels):
    assert len(x) == arr.shape[0]
    for model_ind, model in enumerate(models):
        y = arr[..., model_ind]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=ysem,
            marker="o",
            label=labels[model_ind],
            linestyle="--",
            ms=4,
            color=colors[model_ind],
        )

    ax.set_xticks(x)
    ax.set_ylim(bottom=0.)
    return ax


def main():
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.566, 2.65),
        squeeze=False,
        constrained_layout=True,
        sharex="row"
    )

    models = [
        NonHierarchicalBayesianModel,
        HierarchicalBayesianModel
    ]
    labels = [
        "Non-hierarchical Bayesian\nWilcoxon signed-rank test",
        "Hierarchical Bayesian\n95% Highest Posterior\nDensity Interval (HPDI)\ntesting"
    ]

    # With effect
    src = os.path.join(EXPERIMENTS_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(f"mae: {mae.shape}")

    src = os.path.join(EXPERIMENTS_DIR, "prob.npy")
    prob = np.load(src)
    logger.info(f"prob: {prob.shape}")

    ax = axes[0, 0]
    ax = _power_plot(ax, N_SUBJECTS_SPACE[1:], prob, models, labels)
    ax.axhline(y=.8, linestyle="--", color="r", linewidth=1, xmax=.99)
    ax.set_yticks([.2 * i for i in range(6)])

    # Without effect
    src = os.path.join(EXPERIMENTS_NO_EFFECT_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(f"mae: {mae.shape}")

    src = os.path.join(EXPERIMENTS_NO_EFFECT_DIR, "prob.npy")
    prob = np.load(src)
    logger.info(f"prob: {prob.shape}")

    ax = axes[0, 1]
    ax = _power_plot(ax, N_SUBJECTS_SPACE[1:], prob, models, labels)
    ax.axhline(y=.05, linestyle="--", color="r", linewidth=1, xmax=.99)
    ax.set_yticks([.02 * i for i in range(3)] + [.05])
    ax.set_ylim(top=.104)
    # ax.set_yticks([.04 * i for i in range(3)] + [.05])

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            ax.set_xlabel("Number of participants", fontsize=axis_label_size)
            ax.set_ylabel("")
            if not i:
                if not j: ax.set_ylabel("True positive rate (1 - β)", fontsize=axis_label_size)
                if j: ax.set_ylabel("False positive rate (α)", fontsize=axis_label_size)
            else:
                ax.set_ylabel("Mean Absolute Error", fontsize=axis_label_size)

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

    Hypothesis = "Hypothesis"
    fig.suptitle(r"$\bf{" + Hypothesis + "}$" + ": $\Delta \sim \mathcal{N}(\mu_{\Delta}, \sigma_{\Delta})$, H0: $\mu_{\Delta} \leq 0$ vs H1: $\mu_{\Delta} > 0$", fontsize=axis_label_size)
    ax = axes[0, 0]
    ax.text(1.4, .805, "80% Power", va="bottom", ha="left", fontsize=7)
    ax.set_title("Null hypothesis is false, $\Delta \sim \mathcal{N}(3, 1.5)$", fontsize=axis_label_size - 1)

    ax = axes[0, 1]
    ax.text(1.4, .051, "0.05 Significance level", va="bottom", ha="left", fontsize=7)
    ax.set_title("Null hypothesis is true, $\Delta \sim \mathcal{N}(0, 1.5)$", fontsize=axis_label_size - 1)
    # ax.legend(loc="upper left", fontsize=6)

    ax = axes[0, 1]
    ax.legend(loc="upper center", fontsize=6.5)

    fig.align_xlabels()
    fig.align_ylabels()

    dest = os.path.join(BUILD_DIR, "power.svg")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(BUILD_DIR, "power.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
