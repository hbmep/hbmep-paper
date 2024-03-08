import os
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel
from core__number_of_reps_per_pulse import N_SUBJECTS
from constants import N_PULSES_SPACE, N_REPS_PER_PULSE_SPACE, NUMBER_OF_REPS_PER_PULSE_DIR

logger = logging.getLogger(__name__)

BUILD_DIR = NUMBER_OF_REPS_PER_PULSE_DIR


def main():
    n_subjects = N_SUBJECTS
    n_pulses_space = N_PULSES_SPACE
    n_reps_space = N_REPS_PER_PULSE_SPACE
    M = HierarchicalBayesianModel
    draws_space = range(5)

    mae = []
    mse = []
    for n_reps in n_reps_space:
        for n_pulses in n_pulses_space:
            for draw in draws_space:
                n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
                draw_dir = f"d{draw}"

                match M.NAME:
                    case "hierarchical_bayesian_model":
                        dir = os.path.join(
                            BUILD_DIR,
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

                    case _:
                        raise ValueError(f"Invalid model {M.NAME}.")

                curr_mae = np.abs(a_true - a_pred).mean()
                curr_mse = np.square(a_true - a_pred).mean()
                mae.append(curr_mae)
                mse.append(curr_mse)

    mae = np.array(mae).reshape(len(n_reps_space), len(n_pulses_space), len(draws_space))
    mse = np.array(mse).reshape(len(n_reps_space), len(n_pulses_space), len(draws_space))

    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")

    nrows, ncols = 1, 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 5, nrows * 3),
        squeeze=False,
        constrained_layout=True
    )

    ax = axes[0, 0]
    for reps_ind, n_reps in enumerate(n_reps_space):
        x = n_pulses_space
        y = mae[reps_ind, ...]
        yme = y.mean(axis=-1)
        ysem = stats.sem(y, axis=-1)
        ax.errorbar(
            x=x,
            y=yme,
            yerr=ysem,
            marker="o",
            label=f"reps: {n_reps}",
            linestyle="--",
            ms=4
        )
        ax.set_xticks(x)
        ax.legend(bbox_to_anchor=(0., 1.2), loc="center", fontsize=6)
        ax.set_xlabel("# Pulses")
        ax.set_ylabel("MAE")

    ax.set_title("8 Subjects")
    ax.set_ylim(bottom=0.)

    fig.align_xlabels()
    fig.align_ylabels()

    dest = os.path.join(BUILD_DIR, "results.png")
    fig.savefig(dest, dpi=600)
    logger.info(f"Saved to {dest}")

    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
