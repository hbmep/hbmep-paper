import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import jax

from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import ReLU
from core import (N_REPS, N_PULSES, EXPERIMENT_NAME)

logger = logging.getLogger(__name__)
SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-speed/simulate_data"
EXPERIMENT_DIR = os.path.join(SIMULATION_DIR, EXPERIMENT_NAME)


def main():
    draws_space = range(10)
    methods_space = ["mcmc", "svi_jit"]
    M = ReLU

    """ Results """
    mae = []
    mse = []
    time = []
    for method in methods_space:
        for draw in draws_space:
            draw_dir = f"d{draw}"
            dir = os.path.join(
                EXPERIMENT_DIR,
                draw_dir,
                M.NAME,
                method
            )
            a_true = np.load(os.path.join(dir, "a_true.npy"))
            a_pred = np.load(os.path.join(dir, "a_pred.npy"))

            a_pred = a_pred.mean(axis=0).reshape(-1,)
            a_true = a_true.reshape(-1,)

            curr_mae = np.abs(a_true - a_pred).mean()
            curr_mse = np.square(a_true - a_pred).mean()
            mae.append(curr_mae)
            mse.append(curr_mse)

            curr_time = np.load(os.path.join(dir, "time_taken.npy")).item()
            time.append(curr_time)

    mae = np.array(mae).reshape(len(methods_space), len(draws_space))
    mse = np.array(mse).reshape(len(methods_space), len(draws_space))
    time = np.array(time).reshape(len(methods_space), len(draws_space))

    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")
    logger.info(f"Time: {time.shape}")

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), squeeze=False, constrained_layout=True)

    # ax = axes[0, 0]
    # import seaborn as sns
    # for method_ind, method in enumerate(methods_space):
    #     sns.scatterplot(x=draws_space, y=mae[method_ind, :], label=method, ax=ax)

    # ax = axes[0, 1]
    # sns.barplot(time.T, ax=ax)

    for method_ind, method in enumerate(methods_space):
        y = mae[method_ind, :]
        yme = y.mean()
        ysem = stats.sem(y)
        logger.info(f"MAE: {method} mean:{yme}, sem:{ysem}")
        y = time[method_ind, :]
        yme = y.mean()
        ysem = np.std(y)
        logger.info(f"Time: {method} mean:{yme} sec, sdev:{ysem} sec")

    msg = f"% decrease Time = {(100 * (time[0, :].mean() - time[-1, :].mean())) / time[0, :].mean()}%"
    logger.info(msg)
    msg = f"% increase MAE = {(100 * (mae[0, :].mean() - mae[-1, :].mean())) / mae[0, :].mean()}%"
    logger.info(msg)

    # logger.info(time)

    fig.align_xlabels()
    fig.align_ylabels()
    dest = os.path.join(EXPERIMENT_DIR, "result.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    # dest = os.path.join(EXPERIMENT_DIR, "mae.npy")
    # np.save(dest, mae)
    # logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=EXPERIMENT_DIR,
        fname=os.path.basename(__file__)
    )
    main()
