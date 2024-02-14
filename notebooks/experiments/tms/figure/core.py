from operator import le
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
n_draws_subjects = 2000
n_draws_pulses = 1000
SEM_CONST = 1

n_subjects_space = [1, 4, 8, 16]
n_pulses_space = [32, 40, 48, 56, 64]
models = ["Ours: hbMEP", "nHB", "MLE", "Optimization"]

colors = sns.light_palette("grey", as_cmap=True)(np.linspace(0.3, 1, 3))
colors = ["k"] + colors[::-1].tolist()

axis_label_size = 8

lineplot_kwargs = {
    "marker":"o", "linestyle":"dashed", "ms":3
}
lineplot_kwargs_inset = {
    "marker":"o", "linestyle":"dashed", "ms":3
}
lineplot_kwargs = {
    "marker":"o", "ms":2, "linewidth":1
}
lineplot_kwargs_inset = {
    "marker":"o", "ms":2, "linewidth":1
}
inset_rotation = 0


def main():
    logger.info(f"number of colors: {len(colors)}")
    experiment_name = "number_of_subjects"
    src = os.path.join(MAT_DIR, experiment_name, "mae.npy")
    subjects_mae = np.load(src)

    experiment_name = "number_of_pulses"
    src = os.path.join(MAT_DIR, experiment_name, "mae.npy")
    pulses_mae = np.load(src)

    logger.info(f"Subjects MAE: {subjects_mae.shape}")
    logger.info(f"Pulses MAE: {pulses_mae.shape}")

    subjects_mae = subjects_mae[:, :n_draws_subjects, :]
    pulses_mae = pulses_mae[:, :n_draws_pulses, :]
    logger.info(f"Subjects MAE: {subjects_mae.shape}")
    logger.info(f"Pulses MAE: {pulses_mae.shape}")

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.566, 2.65), constrained_layout=True, squeeze=False)

    ax = axes[0, 0]
    x = n_subjects_space
    mae = subjects_mae
    for model_ind, model in enumerate(models):
        y = mae[..., model_ind]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ysem = ysem * SEM_CONST
        ax.errorbar(x=x, y=yme, yerr=ysem, label=f"{model}", **lineplot_kwargs, color=colors[model_ind])
        ax.set_xticks(x)

        if model_ind == 0:
            # ins = ax.inset_axes([0.1,0.74,0.4,0.22], zorder=1)
            ins = ax.inset_axes([0.555,0.72,0.4,0.25], zorder=1)
            ins.errorbar(x=x, y=yme, yerr=ysem, label=f"{model}", **lineplot_kwargs_inset, color=colors[model_ind])
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
                labelrotation=inset_rotation,
                labelsize=6
            )
            ins.yaxis.set_major_locator(plt.MaxNLocator(2))
            ins.set_yticks([1.7, 2.1, 2.5])
            # ins.set_yticks([1.7, 2.5])
        # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        # ax.set_yticks([1.75, 2.75, 3.75])


    ax = axes[0, 1]
    x = n_pulses_space
    mae = pulses_mae
    for model_ind, model in enumerate(models):
        y = mae[..., model_ind]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ysem = ysem * SEM_CONST
        ax.errorbar(x=x, y=yme, yerr=ysem, label=f"{model}", **lineplot_kwargs, color=colors[model_ind])
        ax.set_xticks(x)

        if model_ind == 0:
            ins = ax.inset_axes([0.555,0.72,0.4,0.25], zorder=1)
            ins.errorbar(x=x, y=yme, yerr=ysem, label=f"{model}", **lineplot_kwargs_inset, color=colors[model_ind])
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
                labelrotation=inset_rotation,
                labelsize=6
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

    ax = axes[0, 1]
    axes[0, 0].sharey(axes[0, 1])
    ax.set_ylim(bottom=0., top=9.2)
    ax.set_yticks(range(0, 9, 2))
    ax.set_xlabel("Number of Pulses", fontsize=axis_label_size)
    ax.tick_params(labelleft=False)

    ax = axes[0, 0]
    # if ax.get_legend() is not None: ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    legend_kwargs = {
        "fontsize": 8,
        "frameon": False,
        "markerscale": .8,
        "handlelength": 1.98,
        "loc": (0.05, .97),
        "ncols": 2,
        "columnspacing": 0.8,
        "reverse": True
    }
    axes[0, 0].legend(handles[2:], labels[2:], **legend_kwargs)
    axes[0, 1].legend(handles[:2], labels[:2], **legend_kwargs)
    # fig.legend(handles, labels, fontsize=8, frameon=False, markerscale=.8, handlelength=1.98, ncols=4, loc=(0.1, .94), reverse=True, columnspacing=1.2)
    # fig.legend(fontsize=7, frameon=False, markerscale=.8, handlelength=1.98, ncols=4, loc=(0, .7), columnspacing=0.8, reverse=True)

    ax.set_xlabel("Number of Participants", fontsize=axis_label_size)
    ax.set_ylabel("Mean Absolute Error $($% MSO$)$", fontsize=axis_label_size)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.align_labels()

    dest = os.path.join(BUILD_DIR, "combined_figure.svg")
    fig.savefig(dest, dpi=600)

    dest = os.path.join(BUILD_DIR, "combined_figure.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname="combined_figure"
    )
    main()