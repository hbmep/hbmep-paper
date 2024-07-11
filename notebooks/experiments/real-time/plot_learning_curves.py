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
import glob
plt.rcParams['svg.fonttype'] = 'none'
# from matplotlib import rcParams
plt.rcParams['font.family'] = 'sans-serif'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


if __name__ == "__main__":

    fig_format = 'svg'
    overwrite = True
    root_dir = None

    csv_list = [
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N30_triple_muscle_a/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N30_ECR_muscle_a/REC_MT_cond_norm_mean_threshold.csv'
    ]

    for csv in csv_list:
        loaded_df = pd.read_csv(csv)

        # Pop the last row and put it in a separate vector
        last_row_vector = loaded_df.iloc[-1].values
        loaded_df = loaded_df.iloc[:-1]


        # Plot each column in the same figure
        plt.figure()

        colors = sns.color_palette('colorblind')
        colors[0] = (208 / 255, 28 / 255, 138 / 255)

        for idx, column in enumerate(loaded_df.columns):
            plt.plot(loaded_df[column], marker='o', linestyle='-', color=colors[idx], label=column)
            plt.axhline(y=last_row_vector[idx], color=colors[idx], linestyle='--', label=f'LastRow {column}')

        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(1)