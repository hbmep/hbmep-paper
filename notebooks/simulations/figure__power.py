import os
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from hbmep_paper.utils import setup_logging
from models__power import (
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
    NelderMeadOptimization
)
from constants__power import (
    N_SUBJECTS_SPACE,
    SIMULATE_DATA_DIR__POWER,
    EXPERIMENTS_WITH_EFFECT_DIR,
    EXPERIMENTS_WITH_NO_EFFECT_DIR
)

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"

BUILD_DIR = SIMULATE_DATA_DIR__POWER
FREQUENTIST_CUTOFF = .05
BAYESIAN_CUTOFF = 0.

axis_label_size = 12
inside_text_size = 8

colors = ["#00ced1", "purple", "gray", "k"]
IF_SKIP = True
IF_SKIP = not IF_SKIP

markersize = 3
linewidth = 1
linestyle = "--"


def _power_plot(ax, x, arr, models, labels, jitter=0.):
    assert len(x) == arr.shape[0]
    for model_ind, model in enumerate(models):
        jit = jitter * model_ind
        if IF_SKIP and model.NAME in [NelderMeadOptimization.NAME, MaximumLikelihoodModel.NAME]:
            continue
        match model.NAME:
            case HierarchicalBayesianModel.NAME:
                y = arr[..., model_ind] < BAYESIAN_CUTOFF
            case NonHierarchicalBayesianModel.NAME | MaximumLikelihoodModel.NAME | NelderMeadOptimization.NAME:
                y = arr[..., model_ind] < FREQUENTIST_CUTOFF
            case _:
                raise ValueError(f"Invalid model")

        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ax.errorbar(
            x=[i + jit for i in x],
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


def main():
    const = 1.25
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(const * 5.1, const * 2.65),
        squeeze=False,
        constrained_layout=True,
        sharex="row"
    )

    models = [
        NelderMeadOptimization,
        MaximumLikelihoodModel,
        NonHierarchicalBayesianModel,
        HierarchicalBayesianModel,
    ]
    labels = [
        "Nelder-Mead method\nOne-sided signed-rank test",
        "Maximum likelihood estimation\nOne-sided signed-rank test",
        "Non-hierarchical Bayesian\nOne-sided signed-rank test",
        "Hierarchical Bayesian\n95% highest density interval\n(HDI) testing",
    ]

    # With effect
    src = os.path.join(EXPERIMENTS_WITH_EFFECT_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(f"mae: {mae.shape}")

    src = os.path.join(EXPERIMENTS_WITH_EFFECT_DIR, "prob.npy")
    prob = np.load(src)
    logger.info(f"prob: {prob.shape}")

    ax = axes[0, 0]
    ax = _power_plot(ax, N_SUBJECTS_SPACE[1:], prob, models, labels)
    ax.axhline(y=.8, linestyle="--", color="r", linewidth=1, xmax=.98)
    ax.set_yticks([.2 * i for i in range(6)])
    ax.set_ylim(top=1.1)

    # Without effect
    src = os.path.join(EXPERIMENTS_WITH_NO_EFFECT_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(f"mae: {mae.shape}")

    src = os.path.join(EXPERIMENTS_WITH_NO_EFFECT_DIR, "prob.npy")
    prob = np.load(src)
    logger.info(f"prob: {prob.shape}")

    ax = axes[0, 1]
    ax = _power_plot(ax, N_SUBJECTS_SPACE[1:], prob, models, labels, jitter=.1)
    ax.axhline(y=.05, linestyle="--", color="r", linewidth=1, xmax=.98)
    ax.set_yticks([.02 * i for i in range(6)] + [.05])
    ax.set_ylim(top=.11)

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            ax.set_xlabel("Number of participants", fontsize=axis_label_size)
            ax.set_ylabel("")
            if not i:
                if not j: ax.set_ylabel("True positive rate\nof detecting shift in threshold", fontsize=axis_label_size)
                if j: ax.set_ylabel("False positive rate\nof detecting shift in threshold", fontsize=axis_label_size)
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
                labelsize=10
            )
            ax.grid(axis="y", linestyle="--", alpha=.25)

    ax = axes[0, 0]
    ax.text(2, .805, "80% Power", va="bottom", ha="left", fontsize=inside_text_size)

    ax = axes[0, 1]
    ax.text(2, .051, "5% Significance level", va="bottom", ha="left", fontsize=inside_text_size)

    ax = axes[0, 1]
    ax.legend(loc="upper right", fontsize=inside_text_size, labelspacing=1.1, reverse=True)

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
