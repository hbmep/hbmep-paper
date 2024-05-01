import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from hbmep.model.utils import Site as site
from hbmep.nn import functional as F

from hbmep_paper.utils import setup_logging
from models import (
    HierarchicalBayesianModel,
    RectifiedLogisticS50
)
from core import SIMULATION_PPD_PATH
from constants import (
    TOTAL_REPS,
    TOTAL_PULSES,
    TOTAL_SUBJECTS,
    EXPERIMENTS_DIR,
    MAX_INTENSITY
)

logger = logging.getLogger(__name__)

BUILD_DIR = EXPERIMENTS_DIR
markersize = 3
linewidth = 1
axis_label_size = 12

EXCLUDE_BAD_DRAWS = [511]

# TOTAL_SUBJECTS = 1
# EXCLUDE_BAD_DRAWS = []


def main():
    draws_space = list(range(2000))
    draws_space = [u for u in draws_space if u not in EXCLUDE_BAD_DRAWS]
    models = [
        HierarchicalBayesianModel,
        RectifiedLogisticS50
    ]

    true_a, pred_a = [], []
    for draw in draws_space:
        for M in models:
            n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{TOTAL_REPS}", f"p{TOTAL_PULSES}", f"n{TOTAL_SUBJECTS}"
            draw_dir = f"d{draw}"

            dir = os.path.join(
                BUILD_DIR,
                draw_dir,
                n_subjects_dir,
                n_reps_dir,
                n_pulses_dir,
                M.NAME
            )
            a_true = np.load(os.path.join(dir, "a_true.npy"))
            a_pred = np.load(os.path.join(dir, "a_pred.npy"))
            a_pred = a_pred.mean(axis=0)

            true_a.append(a_true)
            pred_a.append(a_pred)

    true_a = np.array(true_a)
    true_a = true_a.reshape(len(draws_space), len(models), *true_a.shape[1:])[..., 0]
    logger.info(f"true_a.shape: {true_a.shape}")

    pred_a = np.array(pred_a)
    pred_a = pred_a.reshape(len(draws_space), len(models), *pred_a.shape[1:])[..., 0]
    logger.info(f"pred_a.shape: {pred_a.shape}")

    mae = np.abs(true_a - pred_a)
    mae = np.swapaxes(mae, -1, -2)
    logger.info(f"mae: {mae.shape}")

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        _, simulation_ppd = pickle.load(g)

    simulation_ppd = {u: v[draws_space, ...] for u, v in simulation_ppd.items()}

    saturation = simulation_ppd[site.L] + simulation_ppd[site.H]
    saturation = saturation[:, :TOTAL_SUBJECTS, ...]
    logger.info(f"saturation: {saturation.shape}")

    named_params = [site.a, site.b, site.L, site.ell, site.H]
    response_at_max_intensity = F.rectified_logistic(
        MAX_INTENSITY, *[simulation_ppd[p] for p in named_params]
    )
    response_at_max_intensity = response_at_max_intensity[:, :TOTAL_SUBJECTS, ...]
    logger.info(f"response_at_max_intensity: {response_at_max_intensity.shape}")

    proportion_saturation_observed = response_at_max_intensity / saturation
    proportion_saturation_observed = proportion_saturation_observed[:, :TOTAL_SUBJECTS, ...]
    logger.info(f"proportion_observed: {proportion_saturation_observed.shape}")

    # Bin based on proportions
    bin_width = .1
    min_bin, max_bin = 0, 1
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    n_bins = len(bins) - 1
    bin_labels = [f"{((bins[i]  + bins[i + 1]) / 2)* 100:.0f}" for i in range(n_bins)]
    bin_labels[0] = bin_labels[0].replace("(", "[")
    logger.info(bins)
    logger.info(n_bins)
    logger.info(bin_labels)

    mae = mae.reshape(-1, len(models))
    proportion_saturation_observed = proportion_saturation_observed.reshape(-1,).tolist()

    mae_binned = {u: [] for u in bin_labels}
    for i, prop in enumerate(proportion_saturation_observed):
        for j in range(n_bins):
            if bins[j] <= prop and prop <= bins[j + 1]:
                mae_binned[bin_labels[j]] += mae[i, :].tolist()
                break

    mae_binned = {u: np.array(v).reshape(-1, len(models)) for u, v in mae_binned.items()}
    for u, v in mae_binned.items():
        logger.info(f"{u}: {v.shape}")

    # Plot
    nrows, ncols = 1, 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5, 3.65),
        constrained_layout=True,
        squeeze=False
    )

    ax = axes[0, 0]

    for model_ind, model in enumerate(models):
        me, sem = [], []
        for i in range(n_bins):
            bin = bin_labels[i]

            y = mae_binned[bin][..., model_ind]
            y = np.array(y)
            yme = y.mean()
            ysem = stats.sem(y)
            me.append(yme)
            sem.append(ysem)

        start = 2
        ax.errorbar(
            x=bins[:-1][start:] + (bin_width / 2),
            y=me[start:],
            yerr=sem[start:],
            marker="o",
            label="Error on threshold" if model.NAME == HierarchicalBayesianModel.NAME else "Error on S$_{50}$",
            linestyle="--" if model.NAME == HierarchicalBayesianModel.NAME else "-",
            ms=markersize,
            linewidth=linewidth,
            color="k"
        )

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

    ax.set_xticks(bins[:-1][start:] + (bin_width / 2))
    ax.set_xticklabels(bin_labels[start:], rotation=15)
    ax.set_xlabel("Percentage of saturation observed (% saturation)", fontsize=axis_label_size)
    ax.set_ylabel("Mean absolute error (% MSO)", fontsize=axis_label_size)
    ax.legend(loc="upper right", fontsize=11, reverse=True)

    dest = os.path.join(BUILD_DIR, "saturation.svg")
    fig.savefig(dest, dpi=600)

    dest = os.path.join(BUILD_DIR, "saturation.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
