import os
import pickle
import logging

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from numpyro.diagnostics import hpdi

from hbmep.model import functional as F
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging

logger = logging.getLogger(__name__)
plt.rcParams["svg.fonttype"] = "none"
# plt.style.use('ggplot')

MAT_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/"
PPD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data/example_reps"
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data"
n_draws_reps = 940
SEM_CONST = 1

n_pulses_space = [32, 40, 48, 56, 64]
n_reps_space = [1, 4, 8]
# models = ["Ours: hbMEP", "nHB", "MLE", "Optimization"]

colors = sns.light_palette("grey", as_cmap=True)(np.linspace(0.2, 1, 3))
colors = ["k"] + colors[::-1].tolist()

axis_label_size = 10

lineplot_kwargs = {
    "marker":"o", "ms":3, "linewidth":1.5
}
lineplot_kwargs_inset = {
    "marker":"o", "ms":3, "linewidth":1.5
}

NUM = 5000
true_color = "red"

def main():
    n_reps_space = [1, 4, 8]
    models = {}
    ps = {}
    dfs = {}
    a_trues = {}
    a_preds = {}
    for n_reps in n_reps_space:
        dir = os.path.join(PPD_DIR, f"reps_{n_reps}")
        src = os.path.join(dir, f"inference.pkl")
        with open(src, "rb") as f:
            model, posterior_samples, = pickle.load(f)
        models[n_reps] = model
        ps[n_reps] = posterior_samples
        src = os.path.join(dir, f"a_true.npy")
        a_true = np.load(src)
        src = os.path.join(dir, f"a_pred.npy")
        a_pred = np.load(src)
        logger.info(f"n_reps: {n_reps}, a_true: {a_true.shape}, a_pred: {a_pred.shape}")
        src = os.path.join(dir, f"df.csv")
        df = pd.read_csv(src)
        dfs[n_reps] = df
        a_trues[n_reps] = a_true
        a_preds[n_reps] = a_pred

    model = models[n_reps_space[0]]
    prediction_df = pd.DataFrame(
        np.linspace(0, 100, NUM),
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

    ###########################################################################
    logger.info(f"number of colors: {len(colors)}")
    experiment_name = "number_of_reps"
    src = os.path.join(MAT_DIR, experiment_name, "mae.npy")
    reps_mae = np.load(src)

    logger.info(f"Reps MAE: {reps_mae.shape}")

    reps_mae = reps_mae[:, :n_draws_reps, :]
    logger.info(f"Reps MAE: {reps_mae.shape}")

    ###########################################################################
    fig = plt.figure(figsize=(6, 4))
    subfigs = fig.subfigures(1, 2)
    subfig = subfigs.flat[1]
    axes = subfig.subplots(1, 1, sharex=True, sharey=True, squeeze=False)
    ax = axes[0, 0]
    x = n_pulses_space
    mae = reps_mae
    logger.info(f"MAE: {mae.shape}")
    for reps_ind, n_reps in enumerate(n_reps_space):
        y = mae[reps_ind, ..., 0]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ysem = ysem * SEM_CONST
        if reps_ind == 0:
            y_expected = [1.45, 1.22, 1.04, .9, .8]
            ax.errorbar(x=x, y=y_expected, yerr=ysem, label=f"Expected: Real Time", **lineplot_kwargs, color="red")
        ax.errorbar(x=x, y=yme, yerr=ysem, label=f"Reps: {n_reps}", **lineplot_kwargs, color=colors[reps_ind])
        ax.set_xticks(x)

        if reps_ind == 0:
            ins = ax.inset_axes([0.6,0.65,0.35,0.18], zorder=1)
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
                labelrotation=0,
                labelsize=8
            )
            ins.yaxis.set_major_locator(plt.MaxNLocator(2))
            ins.set_yticks([1.5, 1.9, 2.3])
            # ins.set_yticks([1.5, 2.3])
            # ins.set_yticks([2.2, 1.6])

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
        labelsize=10
    )
    ax.grid(axis="y", linestyle="--", alpha=.25)
    ax.set_ylabel("")

    ax = axes[0, 0]
    if ax.get_legend() is not None: ax.get_legend().remove()
    ax.legend(fontsize=8, frameon=False, markerscale=.8, handlelength=1.98, loc=(.16, .85), ncols=2, columnspacing=0.8, reverse=True)
    ax.set_xlabel("Number of Pulses", fontsize=axis_label_size)
    ax.set_ylabel("Mean Absolute Error $($% MSO$)$", fontsize=axis_label_size)
    ax.set_ylim(bottom=0.)

    subfig.subplots_adjust(left=.15, right=.99, bottom=.15, top=.98, hspace=.4)

    subfig = subfigs.flat[0]
    axes = subfig.subplots(3, 1, sharex=True, sharey=True, squeeze=False)

    params = [site.a, site.b, site.v, site.L, site.ell, site.H]

    for i, ax in enumerate(axes.flat):
        n_reps = n_reps_space[-(i + 1)]
        df = dfs[n_reps]
        posterior_samples = ps[n_reps]
        ppd = ppds[n_reps]
        x = df[model.intensity]
        y = df[model.response[0]]
        sns.scatterplot(x=x, y=y, color=colors[-(i + 2)], ax=ax)
        x = prediction_df[model.intensity].values
        y = ppd[site.obs].mean(axis=0)[..., 0]
        sns.lineplot(x=x, y=y, color=colors[-(i + 2)], ax=ax)
        logger.info(obs_hpdis[n_reps][..., 0].shape)
        ax.fill_between(
            x,
            obs_hpdis[n_reps][0, ...],
            obs_hpdis[n_reps][1, ...],
            color=colors[-(i + 2)],
            alpha=.15
        )
        ins = ax.inset_axes([0.02,0.65,0.3,0.35], zorder=1)
        # ins.errorbar(x=x, y=yme, yerr=ysem, **lineplot_kwargs_inset, color=colors[reps_ind])
        # ins.set_xticks([])
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
            labelsize=8
        )
        samples = posterior_samples[site.a][:, 0, 0]
        ax.axvline(samples.mean(), color=colors[-(i + 2)], ymax=.3, label="Estimated Threshold")
        ax.axvline(a_trues[n_reps].item(), color=true_color, ymax=.3, label="True Threshold")
        match i:
            case 0:
                x_grid = np.linspace(22, 40, 1000)
                xticks = [26, 39]
                ytop = .25
            case 1:
                x_grid = np.linspace(22, 33, 1000)
                xticks = [26, 32]
                ytop = .5
            case 2:
                x_grid = np.linspace(22, 31, 1000)
                xticks = [25, 28]
                ytop = .85

        kde = stats.gaussian_kde(samples)
        density = kde(x_grid)
        ins.plot(x_grid, density, color=colors[-(i + 2)])
        ins.axvline(samples.mean(), color=colors[-(i + 2)], ymax=.8)
        ins.axvline(a_trues[n_reps].item(), color=true_color, ymax=.8)
        # ins.set_xticks([round(samples.mean(), 1), round(a_trues[n_reps].item(), 1)])
        ins.set_xticks(xticks)
        ins.set_ylim(top=ytop)
        if i == 2:
            ins = ax.inset_axes([0.35,0.65,0.35,0.35], zorder=0)
            sns.scatterplot(x=df[model.intensity], y=df[model.response[0]], color=colors[-(i + 2)], ax=ins)
            ins.set_ylim(bottom=-.2, top=0.5)
            ins.set_xlim(left=18, right=35)
            ins.axvline(samples.mean(), color=colors[-(i + 2)], ymax=.8)
            ins.axvline(a_trues[n_reps].item(), color=true_color, ymax=.8)
            ins.tick_params(
                axis='both',
                which='both',
                left=False,
                bottom=False,
                right=False,
                top=False,
                labelleft=False,
                labelbottom=False,
                labelright=False,
                labeltop=False,
                labelrotation=0,
                labelsize=8
            )
            ins.set_ylabel("")
            ins.set_xlabel("")
        # sns.kdeplot(, ax=ins, color=colors[-(i + 2)])
        # x_grid = np.linspace()
        # stats.gaussian_kde(samples)
        # ins.yaxis.set_major_locator(plt.MaxNLocator(2))
        # ins.set_yticks([1.5, 1.9, 2.3])

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
            labelsize=10
        )
        ax.set_ylabel("")
        ax.set_xlabel("")

    ax = axes[-1, 0]
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 2.5, 5.])
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel("Stimulation Intensity (% MSO)", fontsize=axis_label_size)
    ax.set_ylabel("$\mathregular{MEP}$ $\mathregular{Size}_\mathregular{pk-pk}$ ($\mu V$)", fontsize=axis_label_size)

    ax = axes[0, 0]
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    subfig.subplots_adjust(left=.205, right=.98, bottom=.15, top=.98, hspace=.15)


    fig.align_xlabels()
    fig.align_ylabels()
    fig.align_labels()

    dest = os.path.join(BUILD_DIR, "06_sampling_method_extended.svg")
    fig.savefig(dest, dpi=600)

    dest = os.path.join(BUILD_DIR, "06_sampling_method_extended.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname="reps_figure"
    )
    main()
