import os
import pickle
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

from hb_simulate_data import HBSimulator

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
dest = "/home/vishu/logs/ssp-results.log"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(dest, mode="w"),
        logging.StreamHandler()
    ],
    force=True
)


def main():
    model = "nhb_model"
    # model = "hb_model"
    # dir = f"/home/vishu/repos/hbmep-paper/reports/paper/tms/experiments/sparse-subjects/hb-simulate-data/a_random_mean_-1.5_a_random_scale_1/models/{model}"
    dir = f"/home/vishu/repos/hbmep-paper/reports/paper/tms/experiments/sparse-subjects/hb-simulate-data/a_random_mean_-5_a_random_scale_2/models/{model}"
    n_subjects_space = [1, 2, 4, 8, 12]
    draws_space = [i for i in range(0, 30)]
    # draws_space += [i for i in range(60, 160)]

    mae = []
    mse = []
    prob = []

    for n_sub in n_subjects_space:
        for draw in draws_space:
            n_sub_dir, draw_dir = f"nsub_{n_sub}", f"draw_{draw}"
            src = os.path.join(dir, n_sub_dir, draw_dir, "results.pkl")
            with open(src, "rb") as g:
                results, = pickle.load(g)

            a_true = results["a_true"]
            a_pred = results["a_pred"]
            a_pred_map = a_pred.mean(axis=0)
            a_true = a_true.reshape(-1,)
            a_pred_map = a_pred_map.reshape(-1,)
            curr_mae = mean_absolute_error(a_true[:2], a_pred_map[:2])
            curr_mse = mean_squared_error(a_true[:2], a_pred_map[:2])
            # curr_mae = mean_absolute_error(a_true, a_pred_map)
            # curr_mse = mean_squared_error(a_true, a_pred_map)

            if model == "hb_model":
                a_random_mean = results["a_random_mean"]
                a_random_scale = results["a_random_scale"]
                a_random_mean = a_random_mean.reshape(-1,)
                a_random_scale = a_random_scale.reshape(-1,)
                curr_prob = (a_random_mean < 0).mean(axis=0)
            else:
                curr_prob = 0

            logger.info(f"n_sub: {n_sub}, draw: {draw}")
            logger.info(f"MAE: {curr_mae:.2f}, MSE: {curr_mse:.2f}, PROB: {curr_prob:.2f}")
            logger.info("\n")
            mae.append(curr_mae)
            mse.append(curr_mse)
            prob.append(curr_prob)

    mae_arr = np.array(mae).reshape(len(n_subjects_space), len(draws_space))
    mse_arr = np.array(mse).reshape(len(n_subjects_space), len(draws_space))
    prob_arr = np.array(prob).reshape(len(n_subjects_space), len(draws_space))
    logger.info(f"MAE: {mae_arr.shape}")
    logger.info(f"MSE: {mse_arr.shape}")
    logger.info(f"PROB: {prob_arr.shape}")
    logger.info(f"MAE: {mae_arr}")
    logger.info(f"MSE: {mse_arr}")
    logger.info(f"PROB: {prob_arr}")

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), squeeze=False, constrained_layout=True)

    ax = axes[0, 0]
    x = n_subjects_space
    y = mse_arr
    yme = y.mean(axis=-1)
    ysem = stats.sem(y, axis=-1)
    # logger.info(f"ysem: {ysem.shape}")
    ystd = y.std(axis=-1)
    ax.errorbar(x=x, y=yme, yerr=ysem, marker="o", label=f"{model}", linestyle="--", ms=4)
    ax.set_xticks(x)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    x = n_subjects_space
    y = (prob_arr > .95)
    yme = y.mean(axis=-1)
    ysem = stats.sem(y, axis=-1)
    ystd = y.std(axis=-1)
    ax.errorbar(x=x, y=yme, yerr=ystd, marker="o", label=f"{model}", linestyle="--", ms=4)
    ax.set_xticks(x)
    ax.legend(loc="upper right")

    # dest = f"/home/vishu/repos/hbmep-paper/reports/paper/tms/experiments/sparse-subjects/hb-simulate-data/a_random_mean_-1.5_a_random_scale_1/{model}.png"
    dest = f"/home/vishu/repos/hbmep-paper/reports/paper/tms/experiments/sparse-subjects/hb-simulate-data/a_random_mean_-5_a_random_scale_2/{model}.png"
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    main()