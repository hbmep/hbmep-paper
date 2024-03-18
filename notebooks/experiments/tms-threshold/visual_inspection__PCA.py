import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np
from jax.tree_util import tree_flatten
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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
SIMULATE_DATA_DIR_COPY = SIMULATE_DATA_DIR
# SIMULATE_DATA_DIR = "/home/vishu/testing"
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

    num_samples = tree_flatten(simulation_ppd)[0][0].shape[0]
    num_samples = 100

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

    named_params = [
        site.a, site.b,
        site.L, site.ell, site.H,
        site.c_1, site.c_2
    ]

    # Simulate from prior predictive
    logger.info("Simulating from prior predictive ...")
    prior_predictive = simulator.predict(
        df=simulation_df,
        num_samples=num_samples
    )

    # Concatenate all parameters
    params = None
    ppd_params = None
    prior_params = None
    for named_param in named_params:
        param = posterior_samples[named_param]
        ppd_param = simulation_ppd[named_param]
        prior_param = prior_predictive[named_param]

        if params is None:
            params = param
            ppd_params = ppd_param
            prior_params = prior_param
        else:
            params = np.concatenate([params, param], axis=-1)
            ppd_params = np.concatenate([ppd_params, ppd_param], axis=-1)
            prior_params = np.concatenate([prior_params, prior_param], axis=-1)

    logger.info("After concatenation:")
    logger.info(f"Posterior: {params.shape}")
    logger.info(f"PPD: {ppd_params.shape}")
    logger.info(f"Prior: {prior_params.shape}")

    # Flatten across the (num_samples x participants) dimensions
    params = params.reshape(-1, params.shape[-1])
    ppd_params = ppd_params.reshape(-1, ppd_params.shape[-1])
    prior_params = prior_params.reshape(-1, prior_params.shape[-1])

    logger.info("After reshaping:")
    logger.info(f"Posterior: {params.shape}")
    logger.info(f"PPD: {ppd_params.shape}")
    logger.info(f"Prior: {prior_params.shape}")

    # Pipeline for standard scaling and PCA
    pca_kwargs = {
        "n_components": 2,
        "random_state": 0
    }
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(**pca_kwargs))])

    # Reduce dimensionality
    params_embedded = pipeline.fit_transform(params)
    ppd_params_embedded = pipeline.transform(ppd_params)
    prior_params_embedded = pipeline.transform(prior_params)
    pca = pipeline.named_steps['pca']

    logger.info("After PCA:")
    logger.info(f"Posterior: {params_embedded.shape}")
    logger.info(f"PPD: {ppd_params_embedded.shape}")
    logger.info(f"Prior: {prior_params_embedded.shape}")

    logger.info(f"Explained variance: {pca.explained_variance_}")
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logger.info(f"Sum of explained variance ratio: {pca.explained_variance_ratio_.sum()}")

    # Plot
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8, 8),
        constrained_layout=True,
        squeeze=False
    )

    ax = axes[0, 0]
    sns.scatterplot(
        x=prior_params_embedded[:, 0],
        y=prior_params_embedded[:, 1],
        ax=ax,
        label="Prior",
        color="green"
    )
    sns.scatterplot(
        x=params_embedded[:, 0],
        y=params_embedded[:, 1],
        ax=ax,
        label="Real",
        color="orange"
    )
    sns.scatterplot(
        x=ppd_params_embedded[:, 0],
        y=ppd_params_embedded[:, 1],
        ax=ax,
        label="Simulated",
        color="blue"
    )

    ax = axes[0, 1]
    sns.scatterplot(
        x=params_embedded[:, 0],
        y=params_embedded[:, 1],
        ax=ax,
        label="Real",
        color="orange"
    )
    sns.scatterplot(
        x=ppd_params_embedded[:, 0],
        y=ppd_params_embedded[:, 1],
        ax=ax,
        label="Simulated",
        color="blue"
    )


    num_samples = 4000
    params_embedded = params_embedded.reshape(4000, -1, pca_kwargs["n_components"])
    params_ppd_embedded = ppd_params_embedded.reshape(4000, -1, pca_kwargs["n_components"])

    ax = axes[1, 0]
    arr = params_embedded
    for participant in range(arr.shape[1]):
        sns.scatterplot(
            x=arr[:, participant, 0],
            y=arr[:, participant, 1],
            ax=ax,
        )
    ax.set_title("Real: Colored by participant")

    ax = axes[1, 1]
    arr = params_ppd_embedded
    for participant in range(arr.shape[1]):
        sns.scatterplot(
            x=arr[:, participant, 0],
            y=arr[:, participant, 1],
            ax=ax,
        )
    ax.sharex(axes[1, 0])
    ax.sharey(axes[1, 0])
    ax.set_title("Simulated: Colored by participant")

    ax = axes[2, 0]
    arr = params_embedded
    num_uninjured = 12
    sns.scatterplot(
        x=arr[:, :num_uninjured, 0].reshape(-1,),
        y=arr[:, :num_uninjured, 1].reshape(-1,),
        ax=ax,
    )
    ax.sharex(axes[1, 0])
    ax.sharey(axes[1, 0])
    ax.set_title("Uninjured")

    ax = axes[2, 1]
    sns.scatterplot(
        x=arr[:, num_uninjured:, 0].reshape(-1,),
        y=arr[:, num_uninjured:, 1].reshape(-1,),
        ax=ax,
        # facecolors="none",
        # edgecolors="r"
        # alpha=.5
    )
    ax.sharex(axes[1, 0])
    ax.sharey(axes[1, 0])
    ax.set_title("SCI")

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    axes[0, 0].legend(loc="lower right")
    axes[0, 1].legend(loc="upper right")

    dest = os.path.join(BUILD_DIR, "pca.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved PCA plot to {dest}")

    return


if __name__ == "__main__":
    main()
