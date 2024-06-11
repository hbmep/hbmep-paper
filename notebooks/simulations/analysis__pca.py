import os
import pickle
import logging

import pandas as pd
import numpy as np
from jax.tree_util import tree_flatten
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from hbmep.config import Config
from hbmep.model.utils import Site as site
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models__accuracy import HierarchicalBayesianModel
from constants__accuracy import (
    LEARN_POSTERIOR_DIR,
    SIMULATE_DATA_DIR__ACCURACY,
    INFERENCE_FILE,
    SIMULATION_DF
)

logger = logging.getLogger(__name__)

POSTERIOR_PATH = os.path.join(LEARN_POSTERIOR_DIR, INFERENCE_FILE)
SIMULATION_DF_PATH = os.path.join(SIMULATE_DATA_DIR__ACCURACY, SIMULATION_DF)
SIMULATION_PPD_PATH = os.path.join(SIMULATE_DATA_DIR__ACCURACY, INFERENCE_FILE)

BUILD_DIR = SIMULATE_DATA_DIR__ACCURACY


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

    num_samples = tree_flatten(posterior_samples)[0][0].shape[0]

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
        "random_state": 0,
        "svd_solver": "arpack"
    }
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(**pca_kwargs))])

    params_embedded = pipeline.fit_transform(params)
    ppd_params_embedded = pipeline.transform(ppd_params)
    prior_params_embedded = pipeline.transform(prior_params)

    pca = pipeline.named_steps['pca']

    params_embedded = params_embedded.reshape(num_samples, -1, pca_kwargs["n_components"])
    ppd_params_embedded = ppd_params_embedded.reshape(num_samples, -1, pca_kwargs["n_components"])
    prior_params_embedded = prior_params_embedded.reshape(num_samples, -1, pca_kwargs["n_components"])

    dest = os.path.join(BUILD_DIR, "pca.pkl")
    with open(dest, "wb") as f:
        pickle.dump(
            (pca, params_embedded, ppd_params_embedded, prior_params_embedded), f
        )

    logger.info("After PCA:")
    logger.info(f"Posterior: {params_embedded.shape}")
    logger.info(f"PPD: {ppd_params_embedded.shape}")
    logger.info(f"Prior: {prior_params_embedded.shape}")

    logger.info(f"Explained variance: {pca.explained_variance_}")
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logger.info(f"Sum of explained variance ratio: {pca.explained_variance_ratio_.sum()}")
    return


if __name__ == "__main__":
    main()
