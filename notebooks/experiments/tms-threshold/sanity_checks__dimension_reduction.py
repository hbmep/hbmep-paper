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
    # Load simulated dataframe
    src = SIMULATION_DF_PATH
    simulation_df = pd.read_csv(src)

    # Load simulation ppd
    src = SIMULATION_PPD_PATH
    with open(src, "rb") as g:
        simulator, simulation_ppd = pickle.load(g)

    # Set up logging
    simulator._make_dir(BUILD_DIR)
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )

    # Load learnt posterior
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        _, _, posterior_samples = pickle.load(g)

    named_params = [site.a, site.b, site.L, site.ell, site.H]

    params = None
    ppd_params = None
    for named_param in named_params:
        param = posterior_samples[named_param]
        ppd_param = simulation_ppd[named_param]

        # logger.info(f"Posterior {named_param}: {param.shape}")
        # logger.info(f"PPD {named_param}: {ppd_param.shape}")

        # Concatenate
        if params is None:
            params = param
            ppd_params = ppd_param
        else:
            params = np.concatenate([params, param], axis=-1)
            ppd_params = np.concatenate([ppd_params, ppd_param], axis=-1)

    logger.info("After concatenation:")
    logger.info(f"Posterior: {params.shape}")
    logger.info(f"PPD: {ppd_params.shape}")

    # Reduce dimensionality
    # draw = 4
    # params = params[draw, ...]
    # ppd_params = ppd_params[draw, ...]
    # N_DRAWS = 10
    # params = params[:N_DRAWS, ...]
    # params = ppd_params[:N_DRAWS, ...]

    params = params.reshape(-1, params.shape[-1])
    ppd_params = ppd_params.reshape(-1, ppd_params.shape[-1])
    logger.info("After reshaping:")
    logger.info(f"Posterior: {params.shape}")
    logger.info(f"PPD: {ppd_params.shape}")

    combined = np.concatenate([params, ppd_params], axis=0)
    logger.info(f"Combined: {combined.shape}")

    tsne_kwargs = {
        "n_components": 2,
        "random_state": 0,
        "perplexity": 30,
        "verbose": 1,
    }
    tsne = TSNE(**tsne_kwargs)
    combined_embedded = tsne.fit_transform(combined)
    params_embedded = combined_embedded[:params.shape[0], ...]
    ppd_params_embedded = combined_embedded[params.shape[0]:, ...]
    # params_embedded = tsne.fit_transform(params)
    # ppd_params_embedded = tsne.fit_transform(ppd_params)

    logger.info("After t-SNE:")
    logger.info(f"Combined: {combined_embedded.shape}")
    logger.info(f"Posterior: {params_embedded.shape}")
    logger.info(f"PPD: {ppd_params_embedded.shape}")

    dest = os.path.join(BUILD_DIR, "tsne.pkl")
    with open(dest, "wb") as f:
        pickle.dump(
            (tsne, params_embedded, ppd_params_embedded), f)

    nrows, ncols = 1, 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6, 4),
        constrained_layout=True,
        squeeze=False
    )
    ax = axes[0, 0]
    sns.scatterplot(
        x=ppd_params_embedded[:, 0],
        y=ppd_params_embedded[:, 1],
        ax=ax,
        label="Simulated"
    )
    sns.scatterplot(
        x=params_embedded[:, 0],
        y=params_embedded[:, 1],
        ax=ax,
        label="Real"
    )

    dest = os.path.join(BUILD_DIR, "tsne.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved t-SNE plot to {dest}")

    return


if __name__ == "__main__":
    main()
