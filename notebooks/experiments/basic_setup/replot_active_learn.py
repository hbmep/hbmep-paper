import functools
import os
import pickle
import logging

import pandas as pd
import numpy as np
import jax

from hbmep.config import Config
from hbmep.model.utils import Site as site

from models import RectifiedLogistic
from learn_posterior import TOML_PATH, DATA_PATH

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erf

from models import RectifiedLogistic
from utils import run_inference

from scipy.stats import gaussian_kde
from scipy.integrate import nquad
from joblib import Parallel, delayed
from pathlib import Path

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Change this to indicate path to inference.pkl from learn_posterior.py
TOML_PATH = "/home/mcintosh/Local/gitprojects/hbmep-paper/configs/experiments/basic_setup.toml"

import numpy as np
from scipy.stats import gaussian_kde


def main():
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    root_dir = Path(config.BUILD_DIR)
    config.BUILD_DIR = root_dir / 'simulate_data'

    config.MCMC_PARAMS['num_chains'] = 1
    config.MCMC_PARAMS['num_warmup'] = 500
    config.MCMC_PARAMS['num_samples'] = 1000
    seed = dict()
    seed['ix_gen_seed'] = 10
    seed['ix_participant'] = 62
    opt_param = ['a', 'H']  # ['a', 'H']
    N_max = 30
    N_obs = 15  # this is how many enropy calcs to do per every y drawn from x... larger is better
    assert N_obs % 2 != 0, "Better if N_obs is odd."

    simulator = RectifiedLogistic(config=config)
    simulator._make_dir(simulator.build_dir)

    """ Set up logging in build directory """
    dest = os.path.join(simulator.build_dir, "simulate_data.log")
    logging.basicConfig(
        format=FORMAT,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {dest}")

    """ Load learnt posterior """
    src = root_dir / "learn_posterior" / "inference.pkl"
    with open(src, "rb") as g:
        model, mcmc, posterior_samples = pickle.load(g)


if __name__ == "__main__":
    main()
