import os
import pickle
import logging

import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from hbmep_paper.utils import setup_logging
from models__power import (
    HierarchicalBayesianModel,
    DefaultHierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
)
from models__accuracy import LeastSquares
from constants__power import (
    N_SUBJECTS_SPACE,
    SIMULATE_DATA_DIR__POWER,
    EXPERIMENTS_WITH_EFFECT_DIR,
    EXPERIMENTS_WITH_NO_EFFECT_DIR
)
from figure__number_of_subjects_and_pulses import (
    FIG_SIZE_CONST,
    ROTATION,
    TICK_SIZE,
    AXIS_LABEL_SIZE,
    INSIDE_TEXT_SIZE,
    MARKER_SIZE,
    LINE_WIDTH,
    LINE_STYLE,
    GRID_LINE_STYLE,
    GRID_ALPHA,
    MODEL_NAMES_DICT,
    # MODEL_COLORS,
    MODEL_PLOT_KWARGS,
    TIMES_SEM
)

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"
BUILD_DIR = SIMULATE_DATA_DIR__POWER
JITTER = .2
LEVEL_ALPHA = .5
RCOLOR = "r"

MODEL_NAMES_DICT = {
    HierarchicalBayesianModel.NAME: "Hierarchical Bayesian estimation (HBe)\n$95\%$ highest density interval test",
    DefaultHierarchicalBayesianModel.NAME: "Standard hierarchical Bayesian (HB)\nTwo-sided signed-rank test",
    NonHierarchicalBayesianModel.NAME: "Non-hierarchical Bayesian (nHB)\nTwo-sided signed-rank test",
    MaximumLikelihoodModel.NAME: "Maximum likelihood (ML)\nTwo-sided signed-rank test",
    LeastSquares.NAME: "Least squares method (LSM)\nTwo-sided signed-rank test",
}
MODEL_PLOT_KWARGS[HierarchicalBayesianModel.NAME]["linestyle"] = "-"
BLOCK_SIZE = 100


def main():
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            FIG_SIZE_CONST * 5.1,
            FIG_SIZE_CONST * 2.65
        ),
        squeeze=False,
        constrained_layout=True,
        sharex=True
    )

    # With effect
    src = os.path.join(EXPERIMENTS_WITH_EFFECT_DIR, "results.pkl")
    with open(src, "rb") as f:
        (
            reject,
            model_names,
            n_subjects_space,
        ) = pickle.load(f)

    logger.info(f"n_subjects_space: {n_subjects_space}")
    logger.info(f"reject.shape: {reject.shape}")

    subset = [2, 4, 10, 13, 18, 20]
    ind = [nsub in subset for nsub in n_subjects_space]
    n_subjects_space = np.array(n_subjects_space)[ind].tolist()
    reject = reject[..., ind]

    # Plot: With effect
    ax = axes[0, 0]
    for model_ind, model_name in enumerate(model_names):
        x = n_subjects_space
        y = reject[:, model_ind, ...]
        y = y[:(y.shape[0] // BLOCK_SIZE) * BLOCK_SIZE].reshape(-1, BLOCK_SIZE, *y.shape[1:])
        yme = y.mean(axis=(0, 1))
        yerr = TIMES_SEM * stats.sem(y.mean(axis=1), axis=0)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=yerr,
            # marker="o",
            label=MODEL_NAMES_DICT[model_name],
            # linestyle=LINE_STYLE,
            ms=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            # color=MODEL_COLORS[model_name]
            **MODEL_PLOT_KWARGS[model_name]
        )

    # Without effect
    src = os.path.join(EXPERIMENTS_WITH_NO_EFFECT_DIR, "results.pkl")
    with open(src, "rb") as f:
        (
            reject,
            model_names,
            n_subjects_space,
        ) = pickle.load(f)

    logger.info(f"n_subjects_space: {n_subjects_space}")
    logger.info(f"reject.shape: {reject.shape}")

    # Plot: Without effect
    ax = axes[0, 1]
    for model_ind, model_name in enumerate(model_names):
        x = [JITTER * (model_ind - len(model_names) // 2) + n for n in n_subjects_space]
        y = reject[:, model_ind, ...]
        y = y[:(y.shape[0] // BLOCK_SIZE) * BLOCK_SIZE].reshape(-1, BLOCK_SIZE, *y.shape[1:])
        yme = y.mean(axis=(0, 1))
        yerr = TIMES_SEM * stats.sem(y.mean(axis=1), axis=0)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=yerr,
            # marker="o",
            label=MODEL_NAMES_DICT[model_name],
            # linestyle=LINE_STYLE,
            ms=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            # color=MODEL_COLORS[model_name]
            **MODEL_PLOT_KWARGS[model_name]
        )

    for j in range(ncols):
        ax = axes[0, j]
        ax.set_xlabel("")
        ax.set_ylabel("")
        sides = ["top", "right"]
        for side in sides: ax.spines[side].set_visible(False)
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
            labelrotation=ROTATION,
            labelsize=TICK_SIZE
        )
        ax.grid(axis="y", linestyle=GRID_LINE_STYLE, alpha=GRID_ALPHA)

    ax = axes[0, 0]
    ax.set_yticks([.2 * i for i in range(6)])
    ax.set_ylim(top=1.05)
    ax.axhline(y=.8, linestyle="--", color=RCOLOR, linewidth=LINE_WIDTH, alpha=LEVEL_ALPHA, xmin=0.01, xmax=1.)
    ax.text(2, .81, "80% Power", va="bottom", ha="left", fontsize=INSIDE_TEXT_SIZE)
    ax.set_xlabel("Number of participants", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("True positive rate\nof detecting shift in threshold", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks([2, 4, 8, 10, 12, 13, 16, 18, 20])

    ax = axes[0, 1]
    ax.set_yticks([.02 * i for i in range(6)] + [.05])
    ax.set_ylim(top=.105)
    ax.axhline(y=.05, linestyle="--", color=RCOLOR, linewidth=LINE_WIDTH, alpha=LEVEL_ALPHA, xmin=0.01, xmax=1)
    ax.text(1.8, .051, "5%\nSignificance level", va="bottom", ha="left", fontsize=INSIDE_TEXT_SIZE)
    ax.legend(loc="upper right", fontsize=INSIDE_TEXT_SIZE, labelspacing=.8, reverse=True)
    ax.set_xlabel("Number of participants", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("False positive rate", fontsize=AXIS_LABEL_SIZE)

    fig.align_xlabels()
    fig.align_ylabels()

    dest = os.path.join(BUILD_DIR, "power.svg")
    fig.savefig(dest, dpi=600); logger.info(f"Saved to {dest}")
    dest = os.path.join(BUILD_DIR, "power.png")
    fig.savefig(dest, dpi=600); logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
