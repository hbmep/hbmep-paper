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

# NUM_POINTS = 500
markersize = 3
linewidth = 1
linestyle = "--"
axis_label_size = 8

colors = sns.light_palette("grey", as_cmap=True)(np.linspace(0.3, .8, 2))
colors = ["k"] + colors[::-1].tolist()

# max_color, max_alpha = 255, 100
# posterior_color = (204 / max_color, 204 / max_color, 204 / max_color, 15 / max_alpha)
# scatter_color = (179 / max_color, 179 / max_color, 179 / max_color, 100 / max_alpha)
# scatter_edgecolor = (255 / max_color, 255 / max_color, 255 / max_color, 100 / max_alpha)

BUILD_DIR = SIMULATE_DATA_DIR__ACCURACY

# DRAW, SUBJECT_IND = 5, 3
DRAW, SUBJECT_IND = 14, 1
DRAW, SUBJECT_IND = 27, 3
N_PULSES = 64


def main():
    n_reps_space = N_REPS_PER_PULSE_SPACE
    n_pulses_space = N_PULSES_SPACE

    src = os.path.join(NUMBER_OF_REPS_PER_PULSE_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(f"mae.shape: {mae.shape}")

    fig = plt.figure(figsize=(4.566, 3.))
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
    ax.set_ylabel("Mean absolute error on threshold\n(% MSO)", fontsize=axis_label_size)
    subfig.subplots_adjust(left=.15, right=.97, bottom=.15, top=.98, hspace=.4)

    # subfig = subfigs.flat[0]
    # axes = subfig.subplots(
    #     len(n_reps_space), 1, sharex=True, sharey=True, squeeze=False
    # )
    # params = [site.a, site.b, site.L, site.ell, site.H]

    # for (i, ax), (n_reps_ind, n_reps) in zip(enumerate(axes.flat[::-1]), enumerate(n_reps_space)):
    #     df = dfs[n_reps_ind]

    #     x = df[simulator.intensity]
    #     y = df[simulator.response[0]]
    #     sns.scatterplot(x=x, y=y, color=scatter_color, edgecolor=scatter_edgecolor, ax=ax)

    # for i, ax in enumerate(axes.flat):
    #     n_reps = n_reps_space[-(i + 1)]
    #     df = dfs[n_reps]
    #     posterior_samples = ps[n_reps]
    #     ppd = ppds[n_reps]
    #     x = df[model.intensity]
    #     y = df[model.response[0]]
    #     sns.scatterplot(x=x, y=y, color=scatter_color, edgecolor=scatter_edgecolor, ax=ax)
    #     x = prediction_df[model.intensity].values
    #     y = ppd[site.obs].mean(axis=0)[..., 0]
    #     sns.lineplot(x=x, y=y, color=colors[- (i + 1)], ax=ax, linewidth=linewidth)
    #     logger.info(obs_hpdis[n_reps][..., 0].shape)
    #     ax.fill_between(
    #         x,
    #         obs_hpdis[n_reps][0, ...],
    #         obs_hpdis[n_reps][1, ...],
    #         color=posterior_color,
    #         alpha=.2
    #     )
    #     ins = ax.inset_axes([0.02,0.65,0.3,0.35], zorder=1)
    #     ins.tick_params(
    #         axis='both',
    #         which='both',
    #         left=False,
    #         bottom=True,
    #         right=False,
    #         top=False,
    #         labelleft=False,
    #         labelbottom=True,
    #         labelright=False,
    #         labeltop=False,
    #         labelrotation=0,
    #         labelsize=6
    #     )
    #     samples = posterior_samples[site.a][:, 0, 0]
    #     match i:
    #         case 0:
    #             x_grid = np.linspace(28, 40, 1000)
    #             xticks = [30, 40]
    #             ytop = .25
    #         case 1:
    #             x_grid = np.linspace(15, 33, 1000)
    #             xticks = [20, 32]
    #             ytop = .5
    #         case 2:
    #             x_grid = np.linspace(24, 34, 1000)
    #             xticks = [27, 32]
    #             ytop = .85
    #         case 3:
    #             x_grid = np.linspace(26, 33, 1000)
    #             xticks = [28, 32]
    #             ytop = .55

    #     x_grid = np.linspace(18, 40, 1000)
    #     kde = stats.gaussian_kde(samples)
    #     density = kde(x_grid)
    #     ins.plot(x_grid, density, color=colors[-(i + 1)], linewidth=linewidth)
    #     ins.axvline(samples.mean(), color="g", ymax=.8, linestyle=linestyle, linewidth=linewidth)
    #     ins.axvline(a_trues[n_reps].item(), color="red", ymax=.8, linestyle=linestyle, linewidth=linewidth)
    #     # ins.set_xticks(xticks)

    # for i, ax in enumerate(axes.flat):
    #     sides = ["top", "right"]
    #     for side in sides:
    #         ax.spines[side].set_visible(False)
    #     ax.tick_params(
    #         axis='both',
    #         which='both',
    #         left=True,
    #         bottom=True,
    #         right=False,
    #         top=False,
    #         labelleft=False,
    #         labelbottom=False,
    #         labelright=False,
    #         labeltop=False,
    #         labelrotation=15,
    #         labelsize=8
    #     )
    #     ax.set_ylabel("")
    #     ax.set_xlabel("")

    # ax = axes[-1, 0]
    # ax.tick_params(
    #     axis='both',
    #     which='both',
    #     labelleft=True,
    #     labelrotation=15,
    #     labelsize=8
    # )
    # ax.set_xticks([0, 50, 100])
    # # ax.set_yticks([0, 2.5, 5.])
    # ax.tick_params(axis="x", labelbottom=True)
    # ax.set_xlabel("Stimulation intensity (% MSO)", fontsize=axis_label_size)
    # # ax.set_ylabel("$\mathregular{MEP}$ $\mathregular{Size}_\mathregular{pk-pk}$ (mV)", fontsize=axis_label_size)
    # ax.set_ylabel("pk-pk (mV)", fontsize=axis_label_size)

    # for i, ax in enumerate(axes.flat):
    #     ax.axvline(a_trues[n_reps].item(), color="red", ymax=.3, label="True Threshold", linestyle=linestyle, linewidth=linewidth)
    #     n_reps = n_reps_space[-(i + 1)]
    #     posterior_samples = ps[n_reps]
    #     samples = posterior_samples[site.a][:, 0, 0]
    #     ax.axvline(samples.mean(), color="g", ymax=.3, label="Estimated Threshold", linestyle=linestyle, linewidth=linewidth)

    # # pos = (33, 5.5)
    # # text_kwargs = {"fontsize": 7}
    # # ax = axes[0, 0]
    # # ax.text(*pos, "8 reps", **text_kwargs)
    # # ax = axes[1, 0]
    # # ax.text(*pos, "4 reps", **text_kwargs)
    # # ax = axes[2, 0]
    # # ax.text(*pos, "2 reps", **text_kwargs)
    # # ax = axes[3, 0]
    # # ax.text(*pos, "1 rep", **text_kwargs)

    # ax = axes[0, 0]
    # # ax.legend(fontsize=6, loc="upper right")
    # subfig.subplots_adjust(left=.21, right=.90, bottom=.15, top=.98, hspace=.25)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.align_labels()

    dest = os.path.join(BUILD_DIR, "reps.svg")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(BUILD_DIR, "reps.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")


def _main():
    # Load simulated dataframe
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    # Generate nested pulses
    pulses_map = generate_nested_pulses(simulator, simulation_df)

    n_reps_space = N_REPS_PER_PULSE_SPACE
    n_pulses_space = N_PULSES_SPACE
    M = HierarchicalBayesianModel

    dfs = []
    a_pred, a_true = None, None

    for n_reps in n_reps_space:
        dir = os.path.join(
            BUILD_DIR,
            f"d{DRAW}",
            f"n{N_SUBJECTS}",
            f"r{n_reps}",
            f"p{N_PULSES}",
            M.NAME
        )

        pulses = pulses_map[N_PULSES][::n_reps]
        ind = (
            (simulation_df[simulator.features[0]] == SUBJECT_IND) &
            (simulation_df[REP] < n_reps) &
            (simulation_df[simulator.intensity].isin(pulses))
        )
        df = simulation_df[ind].reset_index(drop=True).copy()
        df[simulator.response[0]] = simulation_ppd[site.obs][DRAW, ind, 0]

        ind = df[simulator.response[0]] > 0
        df = df[ind].reset_index(drop=True).copy()

        dfs.append(df)

        a_pred_curr = np.load(os.path.join(dir, "a_pred.npy"))
        a_true_curr = np.load(os.path.join(dir, "a_true.npy"))

        if a_pred is None:
            a_pred = a_pred_curr
            a_true = a_true_curr
        else:
            a_pred = np.concatenate((a_pred, a_pred_curr), axis=-1)
            assert (a_true == a_true_curr).all()

    a_pred_map = a_pred.mean(axis=0)
    logger.info(
        f"n_reps: {n_reps}, a_true: {a_true.shape}, a_pred: {a_pred.shape}, a_pred_map: {a_pred_map.shape}"
    )

    error = np.abs(a_pred_map - a_true)
    flag = (
        (error[..., 0] < error[..., 1])
        & (error[..., 1] < error[..., 2])
        & (error[..., 2] < error[..., 3])
    )
    for subject_ind in range(N_SUBJECTS):
        if flag[subject_ind]:
            logger.info(f"subject_ind {subject_ind}: errors {error[subject_ind, ...]}")
    # return

    src = os.path.join(NUMBER_OF_REPS_PER_PULSE_DIR, "mae.npy")
    mae = np.load(src)
    logger.info(f"mae: {mae.shape}")


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
    # _main()
