import os
import logging

import numpy as np

from hbmep_paper.utils import setup_logging
from models__accuracy import (
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
    NelderMeadOptimization,
)
from core__number_of_subjects import (
    N_REPS,
    N_PULSES
)
from constants__accuracy import (
    NUMBER_OF_SUBJECTS_DIR,
    N_SUBJECTS_SPACE
)

logger = logging.getLogger(__name__)

BUILD_DIR = NUMBER_OF_SUBJECTS_DIR
BUILD_DIR = "/mount/mr_lab/vt2353/repos/hbmep-paper/reports/simulations/accuracy/experiments/number_of_subjects"


def main():
    n_reps = N_REPS
    n_pulses = N_PULSES
    n_subjects_space = N_SUBJECTS_SPACE

    draws_space = range(4000)

    models = [
        NelderMeadOptimization,
        # MaximumLikelihoodModel,
        # NonHierarchicalBayesianModel,
        # HierarchicalBayesianModel
    ]

    mae = []
    mse = []
    for n_subjects in n_subjects_space:
        for draw in draws_space:
            for M in models:
                n_reps_dir, n_pulses_dir, n_subjects_dir = f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}"
                draw_dir = f"d{draw}"

                logger.info(f"n_subjects: {n_subjects}, draw: {draw}, model: {M.NAME}")

                match M.NAME:
                    case "hierarchical_bayesian_model" | "svi_hierarchical_bayesian_model":
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

                    case "non_hierarchical_bayesian_model" | "maximum_likelihood_model":
                        n_subjects_dir = f"n{n_subjects_space[-1]}"
                        a_true, a_pred = [], []

                        for subject in range(n_subjects):
                            sub_dir = f"subject{subject}"
                            dir = os.path.join(
                                BUILD_DIR,
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

                    case "nelder_mead_optimization":
                        dir = os.path.join(
                            BUILD_DIR,
                            draw_dir,
                            f"n{n_subjects_space[-1]}",
                            n_reps_dir,
                            n_pulses_dir,
                            M.NAME
                        )
                        a_true = np.load(os.path.join(dir, "a_true.npy"))[:n_subjects, ...]
                        a_pred = np.load(os.path.join(dir, "a_pred.npy"))[:n_subjects, ...]

                        a_pred = a_pred.reshape(-1,)
                        a_true = a_true.reshape(-1,)

                    case _:
                        raise ValueError(f"Invalid model {M.NAME}.")

                curr_mae = np.abs(a_true - a_pred).mean()
                curr_mse = np.square(a_true - a_pred).mean()
                mae.append(curr_mae)
                mse.append(curr_mse)

    mae = np.array(mae).reshape(len(n_subjects_space), len(draws_space), len(models))
    mse = np.array(mse).reshape(len(n_subjects_space), len(draws_space), len(models))

    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")

    dest = os.path.join(BUILD_DIR, "mae.npy")
    np.save(dest, mae)
    logger.info(f"Saved to {dest}")
    dest = os.path.join(BUILD_DIR, "mse.npy")
    np.save(dest, mse)
    logger.info(f"Saved to {dest}")

    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )
    main()
