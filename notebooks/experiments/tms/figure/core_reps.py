import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep_paper.utils import setup_logging

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"
# plt.style.use('ggplot')

MAT_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/"
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data"
n_draws_reps = 940
SEM_CONST = 1

n_pulses_space = [32, 40, 48, 56, 64]
n_reps_space = [1, 4, 8]
# models = ["Ours: hbMEP", "nHB", "MLE", "Optimization"]

colors = sns.light_palette("grey", as_cmap=True)(np.linspace(0.3, 1, 3))
colors = ["k"] + colors[::-1].tolist()

axis_label_size = 10

lineplot_kwargs = {
    "marker":"o", "ms":3, "linewidth":1.5
}
lineplot_kwargs_inset = {
    "marker":"o", "ms":3, "linewidth":1.5
}


def main():
    logger.info(f"number of colors: {len(colors)}")
    experiment_name = "number_of_reps"
    src = os.path.join(MAT_DIR, experiment_name, "mae.npy")
    reps_mae = np.load(src)

    logger.info(f"Reps MAE: {reps_mae.shape}")

    reps_mae = reps_mae[:, :n_draws_reps, :]
    logger.info(f"Reps MAE: {reps_mae.shape}")

    nrows, ncols = 1, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4, 3), constrained_layout=True, squeeze=False)

    ax = axes[0, 0]
    x = n_pulses_space
    mae = reps_mae
    logger.info(f"MAE: {mae.shape}")
    for reps_ind, n_reps in enumerate(n_reps_space):
        y = mae[reps_ind, ..., 0]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ysem = ysem * SEM_CONST
        ax.errorbar(x=x, y=yme, yerr=ysem, label=f"Reps: {n_reps}", **lineplot_kwargs, color=colors[reps_ind])
        ax.set_xticks(x)

        if reps_ind == 0:
            ins = ax.inset_axes([0.555,0.72,0.4,0.25], zorder=1)
            ins.errorbar(x=x, y=yme, yerr=ysem, **lineplot_kwargs_inset, color=colors[reps_ind])
            ins.set_xticks([])
            ins.tick_params(
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
            ins.yaxis.set_major_locator(plt.MaxNLocator(2))
            ins.set_yticks([1.5, 1.9, 2.3])
            # ins.set_yticks([1.5, 2.3])
            # ins.set_yticks([2.2, 1.6])

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
            labelsize=10
        )
        ax.grid(axis="y", linestyle="--", alpha=.25)
        ax.set_ylabel("")


    ax = axes[0, 0]
    if ax.get_legend() is not None: ax.get_legend().remove()
    ax.legend(fontsize=8, frameon=False, markerscale=.8, handlelength=1.98, loc=(.2, .5), ncols=1, bbox_to_anchor=(0.1, .5, .5, 0.5), columnspacing=0.8, reverse=True)
    ax.set_xlabel("Number of Pulses", fontsize=axis_label_size)
    ax.set_ylabel("Mean Absolute Error $($% MSO$)$", fontsize=axis_label_size)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.align_labels()

    dest = os.path.join(BUILD_DIR, "reps_figure.svg")
    fig.savefig(dest, dpi=600)

    dest = os.path.join(BUILD_DIR, "reps_figure.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname="reps_figure"
    )
    main()
