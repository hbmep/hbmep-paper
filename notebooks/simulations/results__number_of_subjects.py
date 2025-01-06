import os
import pickle
import logging

import numpy as np
from scipy import stats

from hbmep_paper.utils import setup_logging
from models__accuracy import (
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
    LeastSquares,
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


def main(draws_space, n_subjects_space, models):
    n_reps = N_REPS
    n_pulses = N_PULSES

    num_draws_processed = 0
    draws_not_processed = []
    mae, mse = [], []

    for draw in draws_space:
        curr_mae, curr_mse = [], []

        try:
            for M in models:
                for n_subjects in n_subjects_space:
                    n_reps_dir, n_pulses_dir, n_subjects_dir, draw_dir = (
                        f"r{n_reps}", f"p{n_pulses}", f"n{n_subjects}", f"d{draw}"
                    )

                    match M.NAME:
                        case HierarchicalBayesianModel.NAME:
                            src = os.path.join(
                                BUILD_DIR,
                                draw_dir,
                                n_subjects_dir,
                                n_reps_dir,
                                n_pulses_dir,
                                M.NAME
                            )
                            a_true = np.load(os.path.join(src, "a_true.npy"))
                            a_true = a_true.reshape(-1,)
                            a_pred = np.load(os.path.join(src, "a_pred.npy"))
                            a_pred = a_pred.mean(axis=0).reshape(-1,)

                        case NonHierarchicalBayesianModel.NAME | MaximumLikelihoodModel.NAME:
                            src = os.path.join(
                                BUILD_DIR,
                                draw_dir,
                                f"n{n_subjects_space[-1]}",
                                n_reps_dir,
                                n_pulses_dir,
                                M.NAME
                            )
                            a_true = np.load(os.path.join(src, "a_true.npy"))
                            a_true = a_true[:n_subjects, ...]
                            a_true = a_true.reshape(-1,)
                            a_pred = np.load(os.path.join(src, "a_pred.npy"))
                            a_pred = a_pred[:, :n_subjects, ...]
                            a_pred = a_pred.mean(axis=0).reshape(-1,)

                        case LeastSquares.NAME:
                            src = os.path.join(
                                BUILD_DIR,
                                draw_dir,
                                f"n{n_subjects_space[-1]}",
                                n_reps_dir,
                                n_pulses_dir,
                                M.NAME
                            )
                            a_true = np.load(os.path.join(src, "a_true.npy"))
                            a_true = a_true[:n_subjects, ...]
                            a_true = a_true.reshape(-1,)
                            a_pred = np.load(os.path.join(src, "a_pred.npy"))
                            a_pred = a_pred[:n_subjects, ...]
                            a_pred = a_pred.reshape(-1,)

                        case _:
                            raise ValueError(f"Invalid model {M.NAME}.")

                    curr_mae.append(np.abs(a_true - a_pred).mean())
                    curr_mse.append(np.square(a_true - a_pred).mean())


        except FileNotFoundError:
            draws_not_processed.append(draw)
            logger.info(f"Draw: {draw} - Missing")

        else:
            logger.info(f"Draw: {draw}")
            mae += curr_mae
            mse += curr_mse
            num_draws_processed += 1

    mae = np.array(mae)
    mae = mae.reshape(num_draws_processed, len(models), len(n_subjects_space))
    mse = np.array(mse)
    mse = mse.reshape(num_draws_processed, len(models), len(n_subjects_space))

    logger.info(f"MAE: {mae.shape}\n{mae.mean(axis=0)}")
    logger.info(f"\n{mae.mean(axis=0) - 1.96 * stats.sem(mae, axis=0)}")
    logger.info(f"\n{mae.mean(axis=0) + 1.96 * stats.sem(mae, axis=0)}")

    model_names = [M.NAME for M in models]
    dest = os.path.join(BUILD_DIR, "results.pkl")
    with open (dest, "wb") as f:
        pickle.dump(
            (model_names, n_subjects_space, mae, mse), f
        )
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )

    draws_space = range(4000)
    models = [
        LeastSquares,
        MaximumLikelihoodModel,
        NonHierarchicalBayesianModel,
        HierarchicalBayesianModel
    ]
    n_subjects_space = N_SUBJECTS_SPACE

    main(
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        models=models
    )
