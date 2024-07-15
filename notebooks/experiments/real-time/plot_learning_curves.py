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
import re


plt.rcParams['svg.fonttype'] = 'none'
# from matplotlib import rcParams
plt.rcParams['font.family'] = 'sans-serif'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def find_consecutive_true(df, eta, T):
    # Compute the absolute difference ratio
    diff_ratio = df.pct_change().abs()

    # Check if the ratio is less than eta
    condition = diff_ratio < eta

    # Find the index where the condition is true for T consecutive rows
    # np.zeros()
    result = np.zeros(len(condition.columns)) * np.nan
    for ix_col, col in enumerate(condition.columns):
        count = 0
        for i in range(1, len(condition)):
            if condition[col].iloc[i]:
                count += 1
                if count == T:
                    result[ix_col] = i + 1  # the + 1 is because i is zero indexed
            else:
                count = 0

    return result


def find_consecutive_gt(df, eta, T):
    # Compute the absolute difference ratio

    # Check if the ratio is less than eta
    condition = df < eta

    # Find the index where the condition is true for T consecutive rows
    # np.zeros()
    result = np.zeros(len(condition.columns)) * np.nan
    for ix_col, col in enumerate(condition.columns):
        count = 0
        for i in range(1, len(condition)):
            if condition[col].iloc[i]:
                count += 1
                if count == T:
                    result[ix_col] = i + 1  # the + 1 is because i is zero indexed
            else:
                count = 0

    return result


def extract_info_from_paths(csv_list):
    pattern = r'/\w+_(\w+)_muscle.*?ix(\d+)/'
    extracted_info = []
    for path in csv_list:
        match = re.search(pattern, path)
        if match:
            word_before_muscle = match.group(1)
            number_after_ix = int(match.group(2))
            extracted_info.append((word_before_muscle, number_after_ix))
    return extracted_info

if __name__ == "__main__":

    fig_format = 'svg'
    overwrite = True
    root_dir = None

    csv_list = [
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_triple_muscle_a_ix0062/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_ECR_muscle_a_ix0062/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_FCR_muscle_a_ix0062/REC_MT_cond_norm_mean_threshold.csv',
        # '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_APB_muscle_a_ix0062/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_triple_muscle_a_ix0040/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_ECR_muscle_a_ix0040/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_FCR_muscle_a_ix0040/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_APB_muscle_a_ix0040/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_triple_muscle_a_ix0020/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_ECR_muscle_a_ix0020/REC_MT_cond_norm_mean_threshold.csv',
        # '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_FCR_muscle_a_ix0020/REC_MT_cond_norm_mean_threshold.csv',
        # '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_APB_muscle_a_ix0020/REC_MT_cond_norm_mean_threshold.csv',
    ]


    for csv in csv_list:
        loaded_df = pd.read_csv(csv)
        csv_info = extract_info_from_paths([csv])
        loaded_df_lower = pd.read_csv(csv.replace('mean_threshold', 'ci_threshold_lower'))
        loaded_df_upper = pd.read_csv(csv.replace('mean_threshold', 'ci_threshold_upper'))
        # Pop the last row and put it in a separate vector
        last_row_vector = loaded_df.iloc[-1].values
        loaded_df = loaded_df.iloc[:-1]


        # Plot each column in the same figure
        plt.figure()

        colors = sns.color_palette('colorblind')
        # colors[0] = (208 / 255, 28 / 255, 138 / 255)
        colors_dict = dict()
        colors_dict['APB'] = (208 / 255, 28 / 255, 138 / 255)
        colors_dict['ECR'] = colors[0]
        colors_dict['FCR'] = colors[1]
        # ix_stop = find_consecutive_true(loaded_df, 0.02, 3)  # eta in Alavi is 0.001... seems a bit too small?!
        ix_stop = find_consecutive_gt((loaded_df - last_row_vector).abs(), 1.5, 4)  # eta in Alavi is 0.001... seems a bit too small?!
        print(f'ix:{csv_info[0][1]}')
        if csv_info[0][0] == 'triple':
            print(f'Triple: {np.max(ix_stop):0.1f}, {ix_stop}')
        else:
            print(f'{csv_info[0][0] }: {ix_stop[[csv_info[0][0] == x for x in loaded_df.columns.to_list()]][0]:0.1f}')
        for idx, column in enumerate(loaded_df.columns):
            plt.fill_between(loaded_df.index, loaded_df_lower[column], loaded_df_upper[column], color=colors_dict[column],
                             alpha=0.2)
            plt.plot(loaded_df[column], marker='.', linestyle='-', color=colors_dict[column], label=column)
            plt.axhline(y=last_row_vector[idx], color=colors_dict[column], linestyle='--', label=f'GT {column}')
            plt.axvline(x=ix_stop[idx] - 1, color=colors_dict[column], linestyle='--', label=f'GT {column}')

        plt.title(f'{csv_info[0][0]}, ix{csv_info[0][1]}')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 75])
        plt.xlim([0, 40])
        plt.show()

