import os
import pickle
import logging

import numpy as np
from scipy import stats
from numpyro.diagnostics import hpdi

from hbmep_paper.utils import setup_logging
from models__power import (
    HierarchicalBayesianModel,
    DefaultHierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
)
from models__accuracy import LeastSquares
from core__power import (
    N_REPS, N_PULSES, N_SUBJECTS_SPACE
)
from constants__power import (
    EXPERIMENTS_WITH_EFFECT_DIR,
    EXPERIMENTS_WITH_NO_EFFECT_DIR
)

logger = logging.getLogger(__name__)
SIGNIFICANCE_LEVEL = .05


def main(experiments_dir, draws_space, n_subjects_space, models):
    setup_logging(
        dir=experiments_dir,
        fname=os.path.basename(__file__)
    )

    num_draws_processed = 0
    draws_not_processed = []
    reject = []

    for draw in draws_space:
        curr_reject = []

        try:
            for M in models:
                for n_subjects in n_subjects_space:
                    match M.NAME:
                        case HierarchicalBayesianModel.NAME:
                            src = os.path.join(
                                experiments_dir,
                                f"d{draw}",
                                f"n{n_subjects}",
                                f"r{N_REPS}",
                                f"p{N_PULSES}",
                                M.NAME
                            )
                            a_delta_loc = np.load(os.path.join(src, "a_delta_loc.npy"))
                            diff = a_delta_loc

                            hdi = hpdi(diff, prob=1 - SIGNIFICANCE_LEVEL)
                            decision = (hdi[0] > 0) | (hdi[1] < 0)
                            curr_reject.append(decision)

                        case DefaultHierarchicalBayesianModel.NAME:
                            src = os.path.join(
                                experiments_dir,
                                f"d{draw}",
                                f"n{n_subjects}",
                                f"r{N_REPS}",
                                f"p{N_PULSES}",
                                M.NAME
                            )
                            a = np.load(os.path.join(src, "a_pred.npy"))
                            a = a.mean(axis=0)
                            x = a[:n_subjects, 0, 0]
                            y = a[:n_subjects, 1, 0]
                            assert np.isnan(x).sum() == 0
                            assert np.isnan(y).sum() == 0
                            assert x.shape == y.shape
                            assert x.shape[0] == n_subjects

                            pr = stats.wilcoxon(
                                x=x - y, alternative="two-sided", axis=0
                            ).pvalue
                            decision = pr < SIGNIFICANCE_LEVEL
                            curr_reject.append(decision)

                        case (
                            NonHierarchicalBayesianModel.NAME
                            | MaximumLikelihoodModel.NAME
                            | LeastSquares.NAME
                        ):
                            src = os.path.join(
                                experiments_dir,
                                f"d{draw}",
                                f"n{n_subjects_space[-1]}",
                                f"r{N_REPS}",
                                f"p{N_PULSES}",
                                M.NAME
                            )
                            a = np.load(os.path.join(src, "a_pred.npy"))

                            if M.NAME in [NonHierarchicalBayesianModel.NAME, MaximumLikelihoodModel.NAME]:
                                a = a.mean(axis=0)

                            x = a[:n_subjects, 0, 0]
                            y = a[:n_subjects, 1, 0]
                            assert np.isnan(x).sum() == 0
                            assert np.isnan(y).sum() == 0
                            assert x.shape == y.shape
                            assert x.shape[0] == n_subjects

                            pr = stats.wilcoxon(
                                x=x - y, alternative="two-sided", axis=0
                            ).pvalue
                            decision = pr < SIGNIFICANCE_LEVEL
                            curr_reject.append(decision)

                        case _:
                            raise ValueError(f"Unknown model: {M.NAME}")

        except FileNotFoundError:
            draws_not_processed.append(draw)
            logger.info(f"Draw: {draw} - Missing {src}")

        else:
            logger.info(f"Draw: {draw}")
            reject += curr_reject
            num_draws_processed += 1

    reject = np.array(reject)
    reject = reject.reshape(num_draws_processed, len(models), len(n_subjects_space), *reject.shape[1:])
    logger.info(f"reject.shape: {reject.shape}")
    logger.info(f"\n{reject.mean(axis=0)}")

    model_names = [M.NAME for M in models]
    dest = os.path.join(experiments_dir, "results.pkl")
    with open(dest, "wb") as f:
        pickle.dump(
            (
                reject,
                model_names,
                n_subjects_space,
            ),
            f
        )
    logger.info(f"Saved to: {dest}")
    return


if __name__ == "__main__":
    models = [
        LeastSquares,
        MaximumLikelihoodModel,
        NonHierarchicalBayesianModel,
        DefaultHierarchicalBayesianModel,
        HierarchicalBayesianModel,
    ]

    # # Run for the experiments with effect
    # n_subjects_space = N_SUBJECTS_SPACE[1:]
    # n_subjects_space += [10, 13, 18]
    # n_subjects_space = sorted(n_subjects_space)
    # main(EXPERIMENTS_WITH_EFFECT_DIR, range(2000), n_subjects_space, models)

    # Run for the experiments with no effect
    n_subjects_space = N_SUBJECTS_SPACE[1:]
    main(EXPERIMENTS_WITH_NO_EFFECT_DIR, range(2000), n_subjects_space, models)
