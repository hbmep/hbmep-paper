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
from constants import NUMBER_OF_SUJECTS_DIR, EXPERIMENTS_DIR

logger = logging.getLogger(__name__)

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
    cmap = sns.color_palette("hls", 8)
    # colors = cmap(np.linspace(0, .7, 5 * len(models)))[::-1][::5]
    colors = cmap[2:2+5]

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
            label=f"{model.NAME}",
            linestyle="--",
            ms=3,
            # linewidth=1,
            color=colors[model_ind]
        )
        ax.set_xticks(x)
        # ax.legend(bbox_to_anchor=(0., 1.2), loc="center", fontsize=6)
        ax.set_xlabel("# Subjects")
        ax.set_ylabel("MAE")

    ax.set_ylim(bottom=0.)
    ax.set_xlabel("Number of Participants", fontsize=axis_label_size)
    ax.set_ylabel("Mean Absolute Error $($% MSO$)$", fontsize=axis_label_size)

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

    fig.align_xlabels()
    fig.align_ylabels()

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
