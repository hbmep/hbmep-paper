import os
import logging

import arviz as az
import numpy as np
from scipy import stats

from hbmep_paper.utils import setup_logging
from models import (
    HierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
    NelderMeadOptimization
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
    n_subjects_space = N_SUBJECTS_SPACE

    models = [
        NelderMeadOptimization,
        MaximumLikelihoodModel,
        NonHierarchicalBayesianModel,
        HierarchicalBayesianModel,
    ]

    mae = []
    mse = []
    prob = []
    norm = []

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
                        a_random_mean = a_random_mean[:, 0, 0]

                        if n_subjects > 1:
                            # pr = (a_random_mean < 0.).mean()
                            hdi = az.hdi(a_random_mean, hdi_prob=.95)
                            pr = hdi[-1]

                    case "non_hierarchical_bayesian_model" | "maximum_likelihood_model":
                        n_subjects_dir = f"n{n_subjects_space[-1]}"
                        a_true, a_pred = None, None

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

                            a_true_sub, a_pred_sub = None, None

                            for intervention in range(2):
                                intervention_dir = f"inter{intervention}"
                                a_true_sub_inter = np.load(os.path.join(dir, intervention_dir, "a_true.npy"))
                                a_pred_sub_inter = np.load(os.path.join(dir, intervention_dir, "a_pred.npy"))

                                if a_true_sub is None:
                                    a_true_sub = a_true_sub_inter
                                    a_pred_sub = a_pred_sub_inter
                                else:
                                    a_true_sub = np.concatenate([a_true_sub, a_true_sub_inter], axis=-2)
                                    a_pred_sub = np.concatenate([a_pred_sub, a_pred_sub_inter], axis=-2)

                            if a_true is None:
                                a_true = a_true_sub
                                a_pred = a_pred_sub
                            else:
                                a_true = np.concatenate([a_true, a_true_sub], axis=-3)
                                a_pred = np.concatenate([a_pred, a_pred_sub], axis=-3)

                        a_pred_map = a_pred.mean(axis=0)

                        if n_subjects > 1:
                            pr = (
                                stats.wilcoxon(
                                    x=a_pred_map[:, 1, 0] - a_pred_map[:, 0, 0],
                                    alternative="less"
                                )
                                .pvalue
                            )

                        if n_subjects > 2:
                            norm_test = stats.shapiro(
                                a_pred_map[:, 1, 0] - a_pred_map[:, 0, 0]
                            )
                            norm.append(norm_test.pvalue)

                        a_true = a_true.reshape(-1,)
                        a_pred = a_pred_map.reshape(-1,)

                    case "nelder_mead_optimization":
                        n_subjects_dir = f"n{n_subjects_space[-1]}"

                        dir = os.path.join(
                            build_dir,
                            draw_dir,
                            n_subjects_dir,
                            n_reps_dir,
                            n_pulses_dir,
                            M.NAME
                        )
                        a_true = np.load(os.path.join(dir, "a_true.npy"))[:n_subjects, ...]
                        a_pred = np.load(os.path.join(dir, "a_pred.npy"))[:n_subjects, ...]

                        if n_subjects > 1:
                            pr = (
                                stats.wilcoxon(
                                    x=a_pred[:, 1, 0] - a_pred[:, 0, 0],
                                    alternative="less"
                                )
                                .pvalue
                            )

                        if n_subjects > 2:
                            norm_test = stats.shapiro(
                                a_pred[:, 1, 0] - a_pred[:, 0, 0]
                            )
                            norm.append(norm_test.pvalue)

                        a_pred = a_pred.reshape(-1,)
                        a_true = a_true.reshape(-1,)

                    case _:
                        raise ValueError(f"Invalid model {M.NAME}.")

                curr_mae = np.abs(a_true - a_pred).mean()
                curr_mse = np.square(a_true - a_pred).mean()
                mae.append(curr_mae)
                mse.append(curr_mse)
                if n_subjects > 1: prob.append(pr)

    mae = np.array(mae).reshape(len(n_subjects_space), len(draws_space), len(models))
    mse = np.array(mse).reshape(len(n_subjects_space), len(draws_space), len(models))
    prob = np.array(prob).reshape(len(n_subjects_space) - 1, len(draws_space), len(models))
    norm = np.array(norm).reshape(len(n_subjects_space) - 2, len(draws_space), len(models) - 1)

    logger.info(f"MAE: {mae.shape}")
    logger.info(f"MSE: {mse.shape}")
    logger.info(f"Prob: {prob.shape}")
    logger.info(f"Norm: {norm.shape}")

    dest = os.path.join(build_dir, "mae.npy")
    np.save(dest, mae)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(build_dir, "mse.npy")
    np.save(dest, mse)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(build_dir, "prob.npy")
    np.save(dest, prob)
    logger.info(f"Saved to {dest}")

    dest = os.path.join(build_dir, "norm.npy")
    np.save(dest, norm)
    logger.info(f"Saved to {dest}")

    for model_ind, model in enumerate(models[:-1]):
        not_normal = (norm < .05)[-1, :, model_ind].mean()
        logger.info(
            f"{not_normal * 100}% of the draws (threshold differences estimated by {model.NAME}) are not normal."
        )

    return


if __name__ == "__main__":
    # Run for the experiments with effect
    main(EXPERIMENTS_DIR, range(2000))

    # # Run for the experiments without effect
    # main(EXPERIMENTS_NO_EFFECT_DIR, range(2000))
