import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep_paper.utils import setup_logging
from models__accuracy import (
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
    LeastSquares
)
from models__power import DefaultHierarchicalBayesianModel
from constants__accuracy import (
    N_SUBJECTS_SPACE,
    N_PULSES_SPACE,
    SIMULATE_DATA_DIR__ACCURACY,
    NUMBER_OF_SUBJECTS_DIR,
    NUMBER_OF_PULSES_DIR
)

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"

BUILD_DIR = SIMULATE_DATA_DIR__ACCURACY

# Constants
FIG_SIZE_CONST = 1.25
TIMES_SEM = 1

ROTATION = 0
TICK_SIZE = 10
AXIS_LABEL_SIZE = 12
INSIDE_TEXT_SIZE = 8
MARKER_SIZE = 2.5
LINE_WIDTH = 1
LINE_STYLE = "--"
GRID_LINE_STYLE = "--"
GRID_ALPHA = .2

MODEL_NAMES_DICT = {
    HierarchicalBayesianModel.NAME: "Hierarchical Bayesian (HB)",
    NonHierarchicalBayesianModel.NAME: "Non-hierarchical Bayesian (nHB)",
    MaximumLikelihoodModel.NAME: "Maximum likelihood (ML)",
    LeastSquares.NAME: "Least squares method (LSM)"
}

COLORS = sns.color_palette("colorblind")
COLORS2 = [
    (128, 128, 128),
    (128,0,128),
    (128, 128, 0),
    (0, 206, 209),
]
COLORS2 = [(r / 255, g / 255, b / 255) for r, g, b in COLORS2]

MODEL_COLORS = {
    HierarchicalBayesianModel.NAME: COLORS2[3],
    NonHierarchicalBayesianModel.NAME: COLORS2[1],
    MaximumLikelihoodModel.NAME: COLORS[8],
    LeastSquares.NAME: COLORS[-3],
    DefaultHierarchicalBayesianModel.NAME: COLORS2[3],
}
MODEL_PLOT_KWARGS = {}
for model_name in MODEL_COLORS:
    MODEL_PLOT_KWARGS[model_name] = {
        "color": MODEL_COLORS[model_name],
        "linestyle": LINE_STYLE,
        "marker": "o",
    }
BLOCK_SIZE = 100


def main():
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            FIG_SIZE_CONST * 4.566,
            FIG_SIZE_CONST * 2.65
        ),
        squeeze=False,
        constrained_layout=True,
        sharey=True
    )

    # Number of subjects
    src = os.path.join(NUMBER_OF_SUBJECTS_DIR, "results.pkl")
    with open(src, "rb") as f:
        model_names, n_subjects_space, mae, _, = pickle.load(f)

    logger.info(f"n_subjects_space: {n_subjects_space}")
    logger.info(f"mae.shape: {mae.shape}")

    # Plot: Number of subjects
    ax = axes[0, 0]
    for model_ind, model_name in enumerate(model_names):
        x = n_subjects_space
        y = mae[:, model_ind, :]
        yme = y.mean(axis=0)
        yerr = TIMES_SEM * stats.sem(y, axis=0)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=yerr,
            label=MODEL_NAMES_DICT[model_name],
            ms=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            **MODEL_PLOT_KWARGS[model_name]
        )
    ax.set_xticks(n_subjects_space)

    # Number of pulses
    src = os.path.join(NUMBER_OF_PULSES_DIR, "results.pkl")
    with open(src, "rb") as f:
        model_names, n_pulses_space, mae, _, = pickle.load(f)

    logger.info(f"n_pulses_space: {n_pulses_space}")
    logger.info(f"mae.shape: {mae.shape}")

    # Plot: Number of pulses
    ax = axes[0, 1]
    for model_ind, model_name in enumerate(model_names):
        x = n_pulses_space
        y = mae[:, model_ind, :]
        yme = y.mean(axis=0)
        yerr = TIMES_SEM * stats.sem(y, axis=0)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=yerr,
            label=MODEL_NAMES_DICT[model_name],
            ms=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            **MODEL_PLOT_KWARGS[model_name]
        )
    ax.set_xticks(n_pulses_space)

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
    ax.legend(
        loc="upper right",
		fontsize=INSIDE_TEXT_SIZE,
		reverse=True,
		labelspacing=.8,
    )
    ax.set_xlabel("Number of participants", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean absolute error \nof threshold estimation $($% MSO$)$", fontsize=AXIS_LABEL_SIZE)

    ax = axes[0, 1]
    ax.set_ylim(bottom=0., top=17.)
    ax.set_yticks(np.arange(0, 17, 2))
    ax.set_xlabel("Number of stimuli", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelleft=False)

    fig.align_xlabels()
    fig.align_ylabels()

    dest = os.path.join(BUILD_DIR, "accuracy.svg")
    fig.savefig(dest, dpi=600); logger.info(f"Saved to {dest}")
    dest = os.path.join(BUILD_DIR, "accuracy.png")
    fig.savefig(dest, dpi=600); logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
