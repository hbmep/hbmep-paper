import os
import pickle
import logging

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from numpyro.diagnostics import hpdi

from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel
from core__number_of_reps_per_pulse import (
    N_REPS_PER_PULSE_SPACE,
    N_PULSES_SPACE,
    N_SUBJECTS
)
from figure__reps_per_pulse_example import BUILD_DIR
from constants import (
    INFERENCE_FILE,
    NUMBER_OF_REPS_PER_PULSE_DIR
)

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"

NUM_POINTS = 5000
axis_label_size = 8
colors = sns.light_palette("grey", as_cmap=True)(np.linspace(0.4, 1, 2))
colors = ["k"] + colors[::-1].tolist()

# fill_colors = sns.light_palette("grey", as_cmap=True)(np.linspace(0.25, 1, 4))[::-1]
max_color, max_alpha = 255, 100
posterior_color = (204 / max_color, 204 / max_color, 204 / max_color, 15 / max_alpha)
scatter_color = (179 / max_color, 179 / max_color, 179 / max_color, 100 / max_alpha)
scatter_edgecolor = (255 / max_color, 255 / max_color, 255 / max_color, 100 / max_alpha)


def main():
    # n_reps_space = N_REPS_PER_PULSE_SPACE
    n_reps_space = [1, 4, 8]
    n_pulses_space = N_PULSES_SPACE
    M = HierarchicalBayesianModel

    src = os.path.join(NUMBER_OF_REPS_PER_PULSE_DIR, "mae.npy")
    mae = np.load(src)
    mae = mae[[0, 2, 3], ...]

    logger.info(f"mae: {mae.shape}")

    models = {}
    ps = {}
    dfs = {}
    a_trues = {}
    a_preds = {}

    for n_reps in n_reps_space:
        dir = os.path.join(
            BUILD_DIR,
            f"reps_{n_reps}"
        )

        src = os.path.join(dir, INFERENCE_FILE)
        with open(src, "rb") as f:
            model, posterior_samples, = pickle.load(f)

        models[n_reps] = model
        ps[n_reps] = posterior_samples

        src = os.path.join(dir, f"a_true.npy")
        a_true = np.load(src)
        src = os.path.join(dir, f"a_pred.npy")
        a_pred = np.load(src)
        src = os.path.join(dir, f"df.csv")
        df = pd.read_csv(src)

        dfs[n_reps] = df
        a_trues[n_reps] = a_true
        a_preds[n_reps] = a_pred
        logger.info(f"n_reps: {n_reps}, a_true: {a_true.shape}, a_pred: {a_pred.shape}")

    model = models[n_reps_space[0]]
    prediction_df = pd.DataFrame(
        np.linspace(0, 100, NUM_POINTS),
        columns=[model.intensity]
    )
    logger.info(prediction_df.head())
    prediction_df[model.features[0]] = 0

    ppds = {}
    obs_hpdis = {}
    for n_reps in n_reps_space:
        ppds[n_reps] = model.predict(df=prediction_df, posterior_samples=ps[n_reps])
        logger.info(ppds[n_reps][site.obs].shape)
        obs_hpdis[n_reps] = hpdi(ppds[n_reps][site.obs][:, :, 0], 0.95)
        logger.info(obs_hpdis[n_reps].shape)

    fig = plt.figure(figsize=(4.566, 2.65))
    # fig = plt.figure(figsize=(5.5, 4.))
    subfigs = fig.subfigures(1, 2)

    subfig = subfigs.flat[1]
    axes = subfig.subplots(1, 1, sharex=True, sharey=True, squeeze=False)
    ax = axes[0, 0]

    # cmap = sns.color_palette("hls", 8)
    # # colors = cmap(np.linspace(0, .7, 5 * len(models)))[::-1][::5]
    # colors = [cmap[-1]]
    # begin = 3
    # colors += cmap[begin:begin + 4]
    # colors = [colors[2], colors[1], colors[-1], colors[-2], colors[0]]

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
            label=f"{n_reps} Reps" if n_reps != 1 else "1 Rep",
            linestyle="--",
            ms=4,
            color=colors[n_reps_ind],
            zorder=5 if n_reps_ind == 0 else None
        )
        ax.set_xticks(x)
        ax.set_xlabel("# Pulses")
        ax.set_ylabel("MAE")

    ax.set_ylim(bottom=0.)
    ax.set_xlabel("Number of Intensities", fontsize=axis_label_size)
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
    ax.grid(axis="y", linestyle="--", alpha=.25)
    ax.set_ylabel("")

    ax.legend(loc="upper right", reverse=True, fontsize=8)
    ax.set_ylabel("Mean Absolute Error on Threshold (% MSO)", fontsize=axis_label_size)
    subfig.subplots_adjust(left=.09, right=.99, bottom=.15, top=.98, hspace=.4)


    subfig = subfigs.flat[0]
    axes = subfig.subplots(
        len(n_reps_space), 1, sharex=True, sharey=True, squeeze=False
    )
    params = [site.a, site.b, site.L, site.ell, site.H]

    for i, ax in enumerate(axes.flat):
        n_reps = n_reps_space[-(i + 1)]
        df = dfs[n_reps]
        posterior_samples = ps[n_reps]
        ppd = ppds[n_reps]
        x = df[model.intensity]
        y = df[model.response[0]]
        sns.scatterplot(x=x, y=y, color=scatter_color, edgecolor=scatter_edgecolor, ax=ax)
        x = prediction_df[model.intensity].values
        y = ppd[site.obs].mean(axis=0)[..., 0]
        sns.lineplot(x=x, y=y, color=colors[- (i + 1)], ax=ax)
        logger.info(obs_hpdis[n_reps][..., 0].shape)
        ax.fill_between(
            x,
            obs_hpdis[n_reps][0, ...],
            obs_hpdis[n_reps][1, ...],
            color=posterior_color,
            alpha=.2
        )
        ins = ax.inset_axes([0.02,0.65,0.3,0.35], zorder=1)
        ins.tick_params(
            axis='both',
            which='both',
            left=False,
            bottom=True,
            right=False,
            top=False,
            labelleft=False,
            labelbottom=True,
            labelright=False,
            labeltop=False,
            labelrotation=0,
            labelsize=6
        )
        samples = posterior_samples[site.a][:, 0, 0]
        match i:
            case 0:
                x_grid = np.linspace(28, 40, 1000)
                xticks = [30, 40]
                ytop = .25
            case 1:
                x_grid = np.linspace(15, 33, 1000)
                xticks = [20, 32]
                ytop = .5
            case 2:
                x_grid = np.linspace(22, 34, 1000)
                xticks = [25, 32]
                ytop = .85
            case 3:
                x_grid = np.linspace(26, 33, 1000)
                xticks = [28, 32]
                ytop = .55

        # x_grid = np.linspace(18, 40, 1000)
        kde = stats.gaussian_kde(samples)
        density = kde(x_grid)
        ins.plot(x_grid, density, color=colors[-(i + 1)])
        ins.axvline(samples.mean(), color="b", ymax=.8)
        ins.axvline(a_trues[n_reps].item(), color="red", ymax=.8)
        ins.set_xticks(xticks)

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
            labelleft=True,
            labelbottom=False,
            labelright=False,
            labeltop=False,
            labelrotation=15,
            labelsize=8
        )
        ax.set_ylabel("")
        ax.set_xlabel("")

    ax = axes[-1, 0]
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 2.5, 5.])
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel("Stimulation Intensity (% MSO)", fontsize=axis_label_size)
    ax.set_ylabel("$\mathregular{MEP}$ $\mathregular{Size}_\mathregular{pk-pk}$ (mV)", fontsize=axis_label_size)
    ax.set_ylabel("$\mathregular{MEP}$ $\mathregular{Size}_\mathregular{pk-pk}$ (mV)", fontsize=axis_label_size)

    for i, ax in enumerate(axes.flat):
        ax.axvline(a_trues[n_reps].item(), color="red", ymax=.3, label="True Threshold")
        n_reps = n_reps_space[-(i + 1)]
        posterior_samples = ps[n_reps]
        samples = posterior_samples[site.a][:, 0, 0]
        ax.axvline(samples.mean(), color="b", ymax=.3, label="Estimated Threshold")

    ax = axes[0, 0]
    ax.legend(fontsize=6, loc="upper right")
    subfig.subplots_adjust(left=.21, right=.90, bottom=.16, top=.98, hspace=.25)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.align_labels()

    dest = os.path.join(BUILD_DIR, "results.svg")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

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
