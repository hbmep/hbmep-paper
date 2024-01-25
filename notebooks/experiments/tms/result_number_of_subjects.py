import os
import pickle
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import jax

from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel, NonHierarchicalBayesianModel, MaximumLikelihoodModel
from core_number_of_subjects import (N_REPS, N_PULSES, EXPERIMENT_NAME)

logger = logging.getLogger(__name__)
SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/simulate_data"
EXPERIMENT_DIR = os.path.join(SIMULATION_DIR, EXPERIMENT_NAME)


def main():
    n_reps = N_REPS
    n_pulses = N_PULSES
    n_subjects_space = [1, 4, 8, 16]
    draws_space = range(5000)
    models = [HierarchicalBayesianModel, NonHierarchicalBayesianModel, MaximumLikelihoodModel]

    """ Results """
    mae = []
    mse = []
    for n_subjects in n_subjects_space:
        for draw in draws_space:
            for M in models:
                n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
                draw_dir = f"d{draw}"

                if M.NAME in ["hbm"]:
                    dir = os.path.join(
                        EXPERIMENT_DIR,
                        draw_dir,
                        n_subjects_dir,
                        n_reps_dir,
                        n_pulses_dir,
                        M.NAME
                    )
                    a_true = np.load(os.path.join(dir, "a_true.npy"))
                    a_pred = np.load(os.path.join(dir, "a_pred.npy"))

                    a_pred = a_pred.mean(axis=0).reshape(-1,)
                    a_true = a_true.reshape(-1,)

                elif M.NAME in ["nhbm", "mle"]:
                    n_subjects_dir = f"n{n_subjects_space[-1]}"
                    a_true, a_pred = [], []

                    for subject in range(n_subjects):
                        sub_dir = f"subject{subject}"
                        dir = os.path.join(
                            EXPERIMENT_DIR,
                            draw_dir,
                            n_subjects_dir,
                            n_reps_dir,
                            n_pulses_dir,
                            M.NAME,
                            sub_dir
                        )
                        a_true_sub = np.load(os.path.join(dir, "a_true.npy"))
                        a_pred_sub = np.load(os.path.join(dir, "a_pred.npy"))

                        a_pred_sub_map = a_pred_sub.mean(axis=0)
                        a_true_sub = a_true_sub

                        a_true += a_pred_sub_map.reshape(-1,).tolist()
                        a_pred += a_true_sub.reshape(-1,).tolist()

                    a_true = np.array(a_true)
                    a_pred = np.array(a_pred)

                else:
                    raise ValueError

                curr_mae = np.abs(a_true - a_pred).mean()
                curr_mse = np.square(a_true - a_pred).mean()
                mae.append(curr_mae)
                mse.append(curr_mse)

    mae = np.array(mae).reshape(len(n_subjects_space), len(draws_space), len(models))
    mse = np.array(mse).reshape(len(n_subjects_space), len(draws_space), len(models))

    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")

    nrows, ncols = 1, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3), squeeze=False, constrained_layout=True)

    ax = axes[0, 0]
    for model_ind, model in enumerate(models):
        x = n_subjects_space
        y = mae[..., model_ind]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ax.errorbar(x=x, y=yme, yerr=ysem, marker="o", label=f"{model.NAME}", linestyle="--", ms=4)
        ax.set_xticks(x)
        ax.legend(loc="upper right")
        ax.set_xlabel("# Subjects")
        ax.set_ylabel("MAE")
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        # ax.set_yticks([1.75, 2.75, 3.75])

    ax.set_title("48 Pulses, 1 Rep, 5000 Draws")

    fig.align_xlabels()
    fig.align_ylabels()
    dest = os.path.join(EXPERIMENT_DIR, "result.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=EXPERIMENT_DIR,
        fname=os.path.basename(__file__)
    )
    main()
