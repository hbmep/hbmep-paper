import os
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
    NelderMeadOptimization
)
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
markersize = 3
linewidth = 1.0
linestyle = "--"
axis_label_size = 12
inside_text_size = 8


def main():
    const = 1.25
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(const * 4.566, const * 2.65),
        squeeze=False,
        constrained_layout=True
    )

    n_subjects_space = N_SUBJECTS_SPACE
    models = [
        NelderMeadOptimization,
        MaximumLikelihoodModel,
        NonHierarchicalBayesianModel,
        HierarchicalBayesianModel
    ]
    labels = [
        "Nelder-Mead method",
        "Maximum likelihood estimation",
        "Non-hierarchical Bayesian",
        "Hierarchical Bayesian"
    ]

    colors = ["#00ced1", "#ffa500", "#0000ff", "k"]
    colors = ["#00ced1", "gold", "#0000ff", "k"]

    colors = list(sns.color_palette("colorblind", 5))
    colors = colors[:2] + colors[3:]
    colors += ["k"]

    custom_palette = sns.color_palette("colorblind", 10)
    custom_palette = list(custom_palette)
    c_ind = [0, 1, -3]
    colors = [custom_palette[i] for i in c_ind]
    colors += ["k"]


    colors = ["#00ced1", "purple", "gray", "k"]

    src = os.path.join(NUMBER_OF_SUBJECTS_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(mae.shape)

    ax = axes[0, 0]
    for model_ind, model in enumerate(models):
        logger.info(model.NAME)
        x = n_subjects_space
        y = mae[..., model_ind]
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
            color=colors[model_ind]
        )
        ax.set_xticks(x)
        ax.set_xlabel("# Subjects")
        ax.set_ylabel("MAE")

    ax.set_ylim(bottom=0., top=16.)
    ax.set_xlabel("Number of participants", fontsize=axis_label_size)
    ax.set_ylabel("MAE on Threshold $($% MSO$)$", fontsize=axis_label_size)

    n_pulses_space = N_PULSES_SPACE
    src = os.path.join(NUMBER_OF_PULSES_DIR, "mae.npy")
    mae = np.load(src)

    ax = axes[0, 1]
    for model_ind, model in enumerate(models):
        x = n_pulses_space
        y = mae[..., model_ind]
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
            color=colors[model_ind]
        )
        ax.set_xticks(x)
        ax.set_xlabel("# Pulses")
        ax.set_ylabel("MAE")

    ax.set_ylim(bottom=0.)
    ax.set_xlabel("Number of intensities", fontsize=axis_label_size)
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
            labelleft=True if i == 0 else False,
            labelbottom=True,
            labelright=False,
            labeltop=False,
            labelrotation=15,
            labelsize=10
        )
        ax.grid(axis="y", linestyle=linestyle, alpha=.25)
        ax.set_ylabel("")

    ax = axes[0, 0]
    ax.sharey(axes[0, 1])
    ax.legend(loc="upper right", fontsize=inside_text_size, reverse=True)

    ax = axes[0, 0]
    ax.set_ylabel("Mean absolute error\nof threshold estimation $($% MSO$)$", fontsize=axis_label_size)

    fig.align_xlabels()
    fig.align_ylabels()

    dest = os.path.join(BUILD_DIR, "accuracy.svg")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(BUILD_DIR, "accuracy.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
