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

n_subjects_space = [1, 4, 8, 16]
n_pulses_space = [32, 40, 48, 56, 64]
models = ["HB", "nHB", "MLE", "Nelder Mead"]
# palette = plt.cm.rainbow

# palette = sns.light_palette("muted", as_cmap=True)
# colors = palette(np.linspace(0, 1, len(models)))
# alpha = 1
# colors = [(128, 128, 128, alpha), (128, 128, 128, alpha), (128, 128, 128, alpha), (128, 128, 128, alpha)]
# colors = [(a / 255, b / 255, c / 255, alpha) for a, b, c, alpha in colors]
colors = sns.light_palette("grey", as_cmap=True)(np.linspace(0.2, 1, 4))
colors = colors[::-1]
# colors = colors[:12]
# colors = colors[::3]
# colors = sns.color_palette("hls", 4).as_hex()
# colors = plt.cm.jet(np.linspace(0,1,4))
fs = 10
axis_label_size = 10

lineplot_kwargs = {
    "marker":"o", "linestyle":"dashed", "ms":3
}
lineplot_kwargs_inset = {
    "marker":"o", "linestyle":"dashed", "ms":3
}

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
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 3), constrained_layout=True, squeeze=False)

    ax = axes[0, 0]
    x = n_subjects_space
    mae = subjects_mae
    for model_ind, model in enumerate(models):
        y = mae[..., model_ind]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
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
                labelrotation=15,
                labelsize=8
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
                labelrotation=15,
                labelsize=8
            )
            ins.yaxis.set_major_locator(plt.MaxNLocator(2))
            ins.set_yticks([1.5, 1.9, 2.3])
            # ins.set_yticks([1.5, 2.3])
            # ins.set_yticks([2.2, 1.6])

    # ax.set_title(f"8 Subjects, {n_draws_pulses} Draws", fontsize=fs)

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
    ax.set_xlabel("# Pulses", fontsize=axis_label_size)
    ax.tick_params(labelleft=False)
    # ax.legend(fontsize=8, frameon=False, markerscale=.8, handlelength=1.98, bbox_to_anchor=(0,0.03,1,1))
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], fontsize=6.8, loc="upper right", frameon=False)
    # ax.legend(loc='upper center', bbox_to_anchor=(0, 1), fontsize=6.8, ncol=4)

    ax = axes[0, 0]
    if ax.get_legend() is not None: ax.get_legend().remove()
    ax.legend(fontsize=8, frameon=False, markerscale=.8, handlelength=1.98, loc="upper left", ncols=1, bbox_to_anchor=(0.1, .5, .5, 0.5), columnspacing=0.8)
    # ax.legend(fontsize=8, frameon=False, markerscale=.8, numpoints=3, handlelength=3, loc="upper left", ncols=1, bbox_to_anchor=(0.1, .5, .5, 0.5), columnspacing=0.8)

    ax.set_xlabel("# Subjects", fontsize=axis_label_size)
    ax.set_ylabel("Mean Absolute Error $($%$)$", fontsize=axis_label_size)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.align_labels()
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