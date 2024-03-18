import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel
from constants import (
    TOML_PATH,
    REP,
    SIMULATE_DATA_DIR,
    SIMULATION_DF,
    INFERENCE_FILE,
    NUMBER_OF_SUJECTS_DIR
)
from constants import (
    TOML_PATH,
    TOTAL_SUBJECTS,
    TOTAL_PULSES,
    TOTAL_REPS,
    REP,
    LEARN_POSTERIOR_DIR,
    INFERENCE_FILE,
    SIMULATE_DATA_DIR,
    SIMULATION_DF
)

logger = logging.getLogger(__name__)


POSTERIOR_PATH = os.path.join(LEARN_POSTERIOR_DIR, INFERENCE_FILE)
SIMULATE_DATA_DIR = "/home/vishu/testing"
SIMULATION_DF_PATH = os.path.join(SIMULATE_DATA_DIR, SIMULATION_DF)
SIMULATION_PPD_PATH = os.path.join(SIMULATE_DATA_DIR, INFERENCE_FILE)
BUILD_DIR = SIMULATE_DATA_DIR

N_REPS = 1
N_PULSES = 48
N_SUBJECTS_SPACE = [1, 2, 4, 8, 16]


@timing
def main():
    # Load reduced dimensions
    src = os.path.join(SIMULATE_DATA_DIR, "tsne.pkl")
    with open(src, "rb") as f:
        tsne, params_embedded, ppd_params_embedded = pickle.load(f)

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, _ = pickle.load(g)

    # Set up logging
    simulator._make_dir(BUILD_DIR)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )

    params_embedded = params_embedded.reshape(4000, -1, params_embedded.shape[-1])
    ppd_params_embedded = ppd_params_embedded.reshape(4000, -1, ppd_params_embedded.shape[-1])
    logger.info(f"params_embedded: {params_embedded.shape}")
    logger.info(f"ppd_params_embedded: {ppd_params_embedded.shape}")

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 6),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True
    )

    # Real
    ax = axes[0, 0]
    arr = params_embedded.copy()
    start = 3
    for participant in range(arr.shape[1]):
        sns.scatterplot(
            x=arr[:, participant, 0],
            y=arr[:, participant, 1],
            ax=ax,
            label=f"Participant {participant}",
            alpha=.1
        )

    ax.set_title("Real")
    if ax.get_legend():
        ax.get_legend().remove()

    # Simulated
    ax = axes[0, 1]
    arr = ppd_params_embedded.copy()
    for participant in range(19, arr.shape[1]):
    # start = 5
    # for participant in range(start, start + 1):
        sns.scatterplot(
            x=arr[:, participant, 0],
            y=arr[:, participant, 1],
            ax=ax,
            label=f"Participant {participant}",
            # alpha=.1
        )
    # for draw in range(10):
    #     sns.scatterplot(
    #         x=arr[draw, :, 0],
    #         y=arr[draw, :, 1],
    #         ax=ax,
    #         label=f"Draw {draw}",
    #         # alpha=.1
    #     )
    ax.set_title("Simulated")
    if ax.get_legend():
        ax.get_legend().remove()

    # Real MAP
    ax = axes[1, 0]
    arr = params_embedded.copy()
    arr = arr.mean(axis=0)
    for participant in range(arr.shape[0]):
        sns.scatterplot(
            x=arr[participant: participant + 1, 0],
            y=arr[participant: participant + 1, 1],
            ax=ax,
            label=f"Participant {participant}",
        )
    ax.set_title("Simulated")
    if ax.get_legend():
        ax.get_legend().remove()

    # Simulated MAP
    ax = axes[1, 1]
    arr = ppd_params_embedded.copy()
    arr = arr.mean(axis=0)
    for participant in range(arr.shape[0]):
        sns.scatterplot(
            x=arr[participant: participant + 1, 0],
            y=arr[participant: participant + 1, 1],
            ax=ax,
            label=f"Participant {participant}",
        )
    ax.set_title("Simulated")
    if ax.get_legend():
        ax.get_legend().remove()

    dest = os.path.join(BUILD_DIR, "tsne.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved t-SNE plot to {dest}")

    return


if __name__ == "__main__":
    main()
