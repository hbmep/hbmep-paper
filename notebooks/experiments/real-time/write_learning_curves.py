import functools
import os
import pickle
import logging

import pandas as pd
import numpy as np
import re
import jax
from copy import deepcopy

from hbmep.config import Config
from hbmep.model.utils import Site as site

import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erf

from learn_posterior import TOML_PATH
from models_old import RectifiedLogistic
from utils import run_inference

from scipy.stats import gaussian_kde
from scipy.integrate import nquad
from joblib import Parallel, delayed
from pathlib import Path
import matplotlib.gridspec as gridspec
import cv2  # install opencv-python
import os
import arviz as az

import glob
plt.rcParams['svg.fonttype'] = 'none'
# from matplotlib import rcParams
plt.rcParams['font.family'] = 'sans-serif'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def write_learning_curves(root_dir=None, es=''):
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    if root_dir is None:
        root_dir = Path(config.BUILD_DIR)
    # root_dir = Path('/media/hdd2/mcintosh/OneDrive/Other/Desktop/test12')  # TEMP

    config.MCMC_PARAMS['num_chains'] = 1
    config.MCMC_PARAMS['num_warmup'] = 500
    config.MCMC_PARAMS['num_samples'] = 1000

    subdirs = sorted([os.path.join(root_dir, o) for o in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, o)) and 'learn_posterior_rt' in o])

    with open(root_dir / 'participant/inference.pkl', "rb") as g:
        _, _, posterior_samples_gt, _, _ = pickle.load(g)

    mean_threshold_list = []
    ci_threshold_list_lower = []
    ci_threshold_list_upper = []

    for str_dir in subdirs:

        config.BUILD_DIR = Path(str_dir)

        """ Load learnt posterior """
        src = config.BUILD_DIR / "inference.pkl"
        with open(src, "rb") as g:
            model, mcmc, posterior_samples, df = pickle.load(g)

        vec_muscle = [str_muscle.replace('PKPK_', '') for str_muscle in model.response]

        mean_threshold = np.mean(posterior_samples[site.a][:, 0, :], 0)
        ci_threshold = az.hdi(posterior_samples[site.a][:, 0, :])
        mean_threshold_list.append(mean_threshold)
        ci_threshold_list_lower.append(ci_threshold[0][0])
        ci_threshold_list_upper.append(ci_threshold[0][1])


   # ADD GT AS LAST ENTRY!!!
    mean_threshold_list.append(posterior_samples_gt[site.a][0][0])

    p_mean = config.BUILD_DIR.parent / f"REC_MT_cond_norm{es}_mean_threshold.{'csv'}"
    df = pd.DataFrame(mean_threshold_list, columns=vec_muscle)
    df.to_csv(p_mean, index=False)

    p_mean = config.BUILD_DIR.parent / f"REC_MT_cond_norm{es}_ci_threshold_lower.{'csv'}"
    df = pd.DataFrame(ci_threshold_list_lower, columns=vec_muscle)
    df.to_csv(p_mean, index=False)

    p_mean = config.BUILD_DIR.parent / f"REC_MT_cond_norm{es}_ci_threshold_upper.{'csv'}"
    df = pd.DataFrame(ci_threshold_list_upper, columns=vec_muscle)
    df.to_csv(p_mean, index=False)


    return root_dir


if __name__ == "__main__":

    fig_format = 'svg'
    overwrite = True
    root_dir = None

    write_learning_curves(root_dir=root_dir)
