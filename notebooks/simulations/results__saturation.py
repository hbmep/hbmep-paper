import os
import pickle
import logging

import numpy as np

from hbmep_paper.utils import setup_logging
from models__accuracy import (
    HierarchicalBayesianModel,
    RectifiedLogisticS50
)
from constants__saturation import (
    EXPERIMENTS_DIR__SATURATION,
    TOTAL_SUBJECTS,
    TOTAL_PULSES,
    TOTAL_REPS,
)

logger = logging.getLogger(__name__)


def main(experiments_dir, draws_space, n_subjects, models):
    setup_logging(
        dir=experiments_dir,
        fname=os.path.basename(__file__)
    )

    true, pred = [], []
    num_draws_processed = 0
    draws_not_processed = []

    for draw in draws_space:
        curr_true, curr_pred = [], []
        n_reps_dir, n_pulses_dir, n_subjects_dir, draw_dir = (
            f"r{TOTAL_REPS}", f"p{TOTAL_PULSES}", f"n{n_subjects}", f"d{draw}"
        )

        try:
            for M in models:
                src = os.path.join(
                    experiments_dir,
                    draw_dir,
                    n_subjects_dir,
                    n_reps_dir,
                    n_pulses_dir,
                    M.NAME
                )
                a_true = np.load(os.path.join(src, "a_true.npy"))
                a_pred = np.load(os.path.join(src, "a_pred.npy"))
                a_pred = a_pred.mean(axis=0)
                assert a_true.shape == a_pred.shape
                curr_true.append(a_true); curr_pred.append(a_pred);

        except FileNotFoundError:
            draws_not_processed.append(draw)
            logger.info(f"Draw: {draw} - Missing {src}")

        else:
            logger.info(f"Draw: {draw}")
            true += curr_true; pred += curr_pred;
            num_draws_processed += 1

    true = np.array(true)
    true = true.reshape(num_draws_processed, len(models), *true.shape[1:])[..., 0]
    logger.info(f"true.shape: {true.shape}")

    pred = np.array(pred)
    pred = pred.reshape(num_draws_processed, len(models), *pred.shape[1:])[..., 0]
    logger.info(f"pred.shape: {pred.shape}")

    mae = np.abs(true - pred)
    mae = np.swapaxes(mae, -1, -2)
    logger.info(f"mae: {mae.shape}")

    model_names = [m.NAME for m in models]
    draws_processed = [d for d in draws_space if d not in draws_not_processed]
    dest = os.path.join(experiments_dir, "results.pkl")
    with open(dest, "wb") as f:
        pickle.dump(
            (model_names, n_subjects, mae, draws_processed,), f
        )
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    draws_space = list(range(4000))
    n_subjects = TOTAL_SUBJECTS
    models = [
        HierarchicalBayesianModel,
        RectifiedLogisticS50
    ]
    main(EXPERIMENTS_DIR__SATURATION, draws_space, n_subjects, models)
