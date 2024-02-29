import os
import pickle
import logging

import arviz as az
import matplotlib.pyplot as plt

from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import (
    RectifiedLogistic,
    Logistic5,
    Logistic4,
    RectifiedLinear
)
from constants import (
    INFERENCE_FILE,
    BUILD_DIR
)

logger = logging.getLogger(__name__)


@timing
def main():
    models = [
        RectifiedLogistic,
        Logistic5,
        Logistic4,
        RectifiedLinear
    ]
    inference_data = {}
    for M in models:
        src = os.path.join(BUILD_DIR, M.NAME, INFERENCE_FILE)
        with open(src, "rb") as f:
            mcmc = pickle.load(f)[1]
            mcmc = az.from_numpyro(mcmc)
            inference_data[M.NAME] = mcmc

    comp_df = az.compare(inference_data)
    logger.info(comp_df.to_string())

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    az.plot_compare(comp_df, ax=ax)
    dest = os.path.join(BUILD_DIR, "compare.png")
    fig.savefig(dest)
    return


if __name__ == "__main__":
    setup_logging(BUILD_DIR, os.path.basename(__file__))
    main()
