import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from hbmep.model.utils import Site as site
from hbmep import functional as F

from hbmep_paper.utils import setup_logging
from figure__number_of_subjects_and_pulses import AXIS_LABEL_SIZE
from models__accuracy import (
    HierarchicalBayesianModel,
    RectifiedLogisticS50
)
from core__saturation import SIMULATION_PPD_PATH
from constants__saturation import (
    EXPERIMENTS_DIR__SATURATION,
    MAX_INTENSITY
)

logger = logging.getLogger(__name__)
BUILD_DIR = EXPERIMENTS_DIR__SATURATION

MARKERSIZE = 3
LINEWIDTH = 1
AXIS_LABEL_SIZE = 12


def main():
    src = os.path.join(BUILD_DIR, "results.pkl")
    with open(src, "rb") as f:
        model_names, n_subjects, mae, draws_processed = pickle.load(f)

    logger.info(f"mae: {mae.shape}")

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        _, simulation_ppd = pickle.load(g)

    simulation_ppd = {u: v[draws_processed, ...] for u, v in simulation_ppd.items()}

    saturation = simulation_ppd[site.L] + simulation_ppd[site.H]
    saturation = saturation[:, :n_subjects, ...]
    logger.info(f"saturation: {saturation.shape}")

    named_params = [site.a, site.b, site.L, site.ell, site.H]
    response_at_max_intensity = F.rectified_logistic(
        MAX_INTENSITY, *[simulation_ppd[p] for p in named_params]
    )
    response_at_max_intensity = np.array(response_at_max_intensity)
    response_at_max_intensity = response_at_max_intensity[:, :n_subjects, ...]
    logger.info(f"response_at_max_intensity: {response_at_max_intensity.shape}")

    proportion_saturation_observed = response_at_max_intensity / saturation
    logger.info(f"proportion_observed: {proportion_saturation_observed.shape}")

    # Bin based on proportions
    bin_width = .1
    min_bin, max_bin = 0, 1
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    n_bins = len(bins) - 1
    bin_labels = [f"({bins[i] * 100:.0f}, {bins[i + 1] * 100:.0f}]" for i in range(n_bins)]
    logger.info(bins)
    logger.info(n_bins)
    logger.info(bin_labels)

    mae = mae.reshape(-1, *mae.shape[2:])
    proportion_saturation_observed = proportion_saturation_observed.reshape(-1,).tolist()

    mae_binned = {u: [] for u in bin_labels}
    for i, prop in enumerate(proportion_saturation_observed):
        for j in range(n_bins):
            if bins[j] <= prop and prop <= bins[j + 1]:
                mae_binned[bin_labels[j]].append(mae[i, :].tolist())
                break

    mae_binned = {u: np.array(v) for u, v in mae_binned.items()}
    for u, v in mae_binned.items(): logger.info(f"{u}: {v.shape}")

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
    ins = ax.inset_axes([0.3,0.4,0.6 ,0.3])
    for model_ind, model_name in enumerate(model_names):
        me, sem = [], []
        for i in range(n_bins):
            bin = bin_labels[i]

            y = mae_binned[bin][:, model_ind]
            yme = y.mean()
            ysem = stats.sem(y)
            me.append(yme)
            sem.append(ysem)

        start = 1
        ax.errorbar(
            x=bins[:-1][start:] + (bin_width / 2),
            y=me[start:],
            yerr=sem[start:],
            marker="o",
            label=(
                "Error on threshold" if model_name == HierarchicalBayesianModel.NAME
                else "Error on S$_{50}$"
            ),
            linestyle="--" if model_name == HierarchicalBayesianModel.NAME else "-",
            ms=MARKERSIZE,
            linewidth=LINEWIDTH,
            color="k"
        )

        start_ins = 0
        ins.errorbar(
            x=bins[:-1][start_ins:] + (bin_width / 2),
            y=me[start_ins:],
            yerr=sem[start_ins:],
            marker="o",
            label=(
                "Error on threshold" if model_name == HierarchicalBayesianModel.NAME
                else "Error on S$_{50}$"
            ),
            linestyle="--" if model_name == HierarchicalBayesianModel.NAME else "-",
            ms=MARKERSIZE,
            linewidth=LINEWIDTH,
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
    ax.set_xticklabels(bin_labels[start:], rotation=25)
    ax.set_xlabel("Percentage of saturation observed (% saturation)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean absolute error (% MSO)", fontsize=AXIS_LABEL_SIZE)
    ax.legend(loc="upper right", fontsize=11, reverse=True, frameon=False)
    ax.set_yticks([20 * i for i in range(6)] + [5])
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
        labelsize=6
    )
    ins.grid(axis="y", linestyle="--", alpha=.25)
    ins.set_ylabel("")

    ins.set_xticks(bins[:-1][start_ins:] + (bin_width / 2))
    ins.set_xticklabels(bin_labels[start_ins:], rotation=25)
    ins.set_yticks([0, 200, 400, 800])

    dest = os.path.join(BUILD_DIR, "saturation.svg")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
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
