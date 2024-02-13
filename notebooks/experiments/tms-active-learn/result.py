import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import jax

from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import ActiveReLU, ReLU

logger = logging.getLogger(__name__)
SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-active-learn/simulate_data"
SIMULATION_DF_PATH = os.path.join(SIMULATION_DIR, "simulation_df.csv")
SIMULATION_PPD_PATH = os.path.join(SIMULATION_DIR, "simulation_ppd.pkl")
EXPERIMENT_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-active-learn/simulate_data/relu-active"


def main():
    draws_space = range(4)
    methods_space = ["mcmc"]
    # draws_space = range(5)
    # methods_space = ["mcmc"]
    M = ActiveReLU

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

            curr_time = np.load(os.path.join(dir, "time_taken.npy"))
            logger.info(f"Time taken: {curr_time}")
            time += curr_time.tolist()

    # logger.info(f"MAE: {len(mae)}")
    # logger.info(f"MSE: {len(mse)}")
    # logger.info(f"Time: {len(time)}")
    mae = np.array(mae).reshape(len(methods_space), len(draws_space))
    mse = np.array(mse).reshape(len(methods_space), len(draws_space))
    time = np.array(time).reshape(len(methods_space), len(draws_space), 2)
    logger.info(time)
    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")
    logger.info(f"Time: {time.shape}")

    for method_ind, method in enumerate(methods_space):
        msg = f"MAE: {method} mean:{mae[method_ind, ...].mean()} sem:{stats.sem(mae[method_ind, ...])}"
        logger.info(msg)
        msg = f"Pre Time: {method} mean: {time[method_ind, ..., 0].mean():.3f} sec, sdev:{np.std(time[method_ind, ..., 0]):.3f} sec"
        logger.info(msg)
        msg = f"Post Time: {method} mean: {time[method_ind, ..., 1].mean():.3f} sec, sdev:{np.std(time[method_ind, ..., 1]):.3f} sec"
        logger.info(msg)
    return


if __name__ == "__main__":
    setup_logging(
        dir=EXPERIMENT_DIR,
        fname=os.path.basename(__file__)
    )
    main()
