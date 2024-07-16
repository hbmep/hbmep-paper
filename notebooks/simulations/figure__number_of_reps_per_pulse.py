import os
import pickle
import logging

import pandas as pd
import numpy as np
from pyparsing import line
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from numpyro.diagnostics import hpdi

from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models__accuracy import HierarchicalBayesianModel
from core__number_of_reps_per_pulse import (
    SIMULATION_DF_PATH,
    SIMULATION_PPD_PATH,
    N_REPS_PER_PULSE_SPACE,
    N_PULSES_SPACE,
    N_SUBJECTS,
)
from utils import generate_nested_pulses
from constants__accuracy import (
    SIMULATE_DATA_DIR__ACCURACY,
    NUMBER_OF_REPS_PER_PULSE_DIR,
    REP,
    INFERENCE_FILE,
)

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"

markersize = 3
linewidth = 1
linestyle = "--"
axis_label_size = 8

colors = sns.light_palette("grey", as_cmap=True)(np.linspace(0.3, .8, 2))
colors = ["k"] + colors[::-1].tolist()

max_color, max_alpha = 255, 100
posterior_color = (204 / max_color, 204 / max_color, 204 / max_color, 15 / max_alpha)
scatter_color = (179 / max_color, 179 / max_color, 179 / max_color, 100 / max_alpha)
scatter_edgecolor = (255 / max_color, 255 / max_color, 255 / max_color, 100 / max_alpha)

BUILD_DIR = SIMULATE_DATA_DIR__ACCURACY


def main():
    n_reps_space = N_REPS_PER_PULSE_SPACE
    n_pulses_space = N_PULSES_SPACE

    src = os.path.join(NUMBER_OF_REPS_PER_PULSE_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(f"mae.shape: {mae.shape}")

    const = 1
    fig = plt.figure(figsize=(const * 4.566, const * 2.65))
    subfigs = fig.subfigures(1, 2)

    subfig = subfigs.flat[1]
    axes = subfig.subplots(1, 1, sharex=True, sharey=True, squeeze=False)
    ax = axes[0, 0]

    for n_reps_ind, n_reps in enumerate(n_reps_space):
        x = n_pulses_space
        y = mae[n_reps_ind, ...]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=ysem,
            marker="o",
            label=f"{n_reps} reps" if n_reps != 1 else "1 rep",
            linestyle=linestyle,
            ms=markersize,
            linewidth=linewidth,
            color=colors[n_reps_ind],
            zorder=5 if n_reps_ind == 0 else None
        )
        ax.set_xticks(x)
        ax.set_xlabel("# Pulses")
        ax.set_ylabel("MAE")

    ax.set_ylim(bottom=0.)
    ax.set_xlabel("Number of intensities", fontsize=axis_label_size)
    ax.set_ylabel("MAE on Threshold $($% MSO$)$", fontsize=axis_label_size)

    ax = axes[0, 0]
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
    ax.grid(axis="y", linestyle=linestyle, alpha=.25)
    ax.set_ylabel("")

    ax.legend(loc="upper right", reverse=True, fontsize=8)
    ax.set_ylabel("Mean absolute error \nof threshold estimation (% MSO)", fontsize=axis_label_size)
    subfig.subplots_adjust(left=.15, right=.97, bottom=.15, top=.98, hspace=.4)

    subfig = subfigs.flat[0]
    axes = subfig.subplots(
        len(n_reps_space), 1, sharex=True, sharey=True, squeeze=False
    )
    for i, ax in enumerate(axes.flat):
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
            labelleft=False,
            labelbottom=False,
            labelright=False,
            labeltop=False,
            labelrotation=15,
            labelsize=8
        )
        ax.set_ylabel("")
        ax.set_xlabel("")

    ax = axes[-1, 0]
    ax.tick_params(
        axis='both',
        which='both',
        labelleft=True,
        labelrotation=15,
        labelsize=8
    )
    ax.set_xlabel("Stimulation intensity (% MSO)", fontsize=axis_label_size)
    ax.set_ylabel("pk-pk (mV)", fontsize=axis_label_size)

    ax = axes[0, 0]
    subfig.subplots_adjust(left=.21, right=.90, bottom=.15, top=.98, hspace=.25)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.align_labels()

    dest = os.path.join(BUILD_DIR, "repetitions.svg")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(BUILD_DIR, "repetitions.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
