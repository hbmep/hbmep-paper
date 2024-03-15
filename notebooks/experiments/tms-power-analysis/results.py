import os
import logging

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep_paper.utils import setup_logging
from models import (
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel
)
from core import (
    N_REPS, N_PULSES, N_SUBJECTS_SPACE
)
from constants import (
    EXPERIMENTS_DIR,
    EXPERIMENTS_NO_EFFECT_DIR
)

logger = logging.getLogger(__name__)


def main(build_dir, draws_space):
    setup_logging(
        dir=build_dir,
        fname=os.path.basename(__file__)
    )

    n_reps = N_REPS
    n_pulses = N_PULSES
    n_subjects_space = N_SUBJECTS_SPACE[1:]

    models = [
        NonHierarchicalBayesianModel,
        HierarchicalBayesianModel
    ]

    mae = []
    mse = []
    prob = []
    for n_subjects in n_subjects_space:
        for draw in draws_space:
            for M in models:
                n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
                draw_dir = f"d{draw}"

                match M.NAME:
                    case "hierarchical_bayesian_model":
                        dir = os.path.join(
                            build_dir,
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

                        a_random_mean = np.load(os.path.join(dir, "a_random_mean.npy"))
                        pr = (a_random_mean < 0).mean()

                    case "non_hierarchical_bayesian_model" | "maximum_likelihood_model":
                        n_subjects_dir = f"n{n_subjects_space[-1]}"
                        a_true, a_pred, diff = [], [], []

                        for subject in range(n_subjects):
                            sub_dir = f"subject{subject}"
                            dir = os.path.join(
                                build_dir,
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

                            diff_sub = a_pred_sub_map[0, 1, 0] - a_pred_sub_map[0, 0, 0]
                            diff.append(diff_sub)

                            a_true += a_pred_sub_map.reshape(-1,).tolist()
                            a_pred += a_true_sub.reshape(-1,).tolist()

                        a_true = np.array(a_true)
                        a_pred = np.array(a_pred)

                        pr = stats.wilcoxon(diff, alternative="less").pvalue

                    case "nelder_mead_optimization":
                        dir = os.path.join(
                            build_dir,
                            draw_dir,
                            f"n{n_subjects_space[-1]}",
                            n_reps_dir,
                            n_pulses_dir,
                            M.NAME
                        )
                        a_true = np.load(os.path.join(dir, "a_true.npy"))[:n_subjects, ...]
                        a_pred = np.load(os.path.join(dir, "a_pred.npy"))[:n_subjects, ...]

                        diff = a_pred[:, 1, 0] - a_pred[:, 0, 0]
                        pr = stats.wilcoxon(diff, alternative="less").pvalue

                        a_pred = a_pred.reshape(-1,)
                        a_true = a_true.reshape(-1,)

                    case _:
                        raise ValueError(f"Invalid model {M.NAME}.")

                curr_mae = np.abs(a_true - a_pred).mean()
                curr_mse = np.square(a_true - a_pred).mean()
                mae.append(curr_mae)
                mse.append(curr_mse)
                prob.append(pr)

    mae = np.array(mae).reshape(len(n_subjects_space), len(draws_space), len(models))
    mse = np.array(mse).reshape(len(n_subjects_space), len(draws_space), len(models))
    prob = np.array(prob).reshape(len(n_subjects_space), len(draws_space), len(models))

    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")
    logger.info(f"Prob: {prob.shape}")

    dest = os.path.join(build_dir, "mae.npy")
    np.save(dest, mae)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(build_dir, "mse.npy")
    np.save(dest, mse)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(build_dir, "prob.npy")
    np.save(dest, prob)
    logger.info(f"Saved to {dest}")

    return


if __name__ == "__main__":
    # Run for the experiments with effect
    main(EXPERIMENTS_DIR, range(500))

    # Run for the experiments without effect
    main(EXPERIMENTS_NO_EFFECT_DIR, range(1000))
