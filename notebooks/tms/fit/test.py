import os
import pickle
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep.config import Config
from hbmep.nn import functional as F
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging, run_svi
from models import (
    # MixtureModel,
    RectifiedLogistic,
    Logistic5,
    Logistic4,
    ReLU
)

logger = logging.getLogger(__name__)
LEVEL = logging.INFO


def main():
    nrows, ncols = 1, 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 6, nrows * 6),
        constrained_layout=True,
        squeeze=False
    )
    ax = axes[0, 0]
    params = {
        site.a: 5,
        site.b: 1,
        site.v: 1,
        site.L: 1,
        site.ell: 1,
        site.H: 1
    }
    x = np.linspace(0, 10, 1000)
    y = F.rectified_logistic(x, *params.values())
    sns.lineplot(x=x, y=y, ax=ax)
    dest = "/home/vishu/testing/rectified_logistic.png"
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")


if __name__ == "__main__":
    setup_logging(dir="/home/vishu/testing", fname="test")
    main()
