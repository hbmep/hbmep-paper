import os
import pickle
import logging

import numpy as np
from scipy import stats
from numpyro.diagnostics import hpdi

from hbmep_paper.utils import setup_logging
from bootstrap__models import (
    HierarchicalBayesianModel,
    DefaultHierarchicalBayesianModel,
    NonHierarchicalBayesianModel,
    MaximumLikelihoodModel,
    LeastSquares,
)
from constants import (
    BOOTSTRAP_EXPERIMENTS_NO_EFFECT_DIR,
    BUILD_DIR as BUILD_DIR_PARENT,
    BOOTSTRAP_DIR,
    BOOTSTRAP_EXPERIMENTS_DIR,
    BOOTSTRAP_FILE,
    N_SUBJECTS_SPACE,
)

logger = logging.getLogger(__name__)
BUILD_DIR = BOOTSTRAP_DIR
SIGNIFICANCE_LEVEL = .05


def main(
	draws_space,
	n_subjects_space,
	models,
    no_effect,
    correction_BonferroniHolm=True
):
    experiments_dir = BOOTSTRAP_EXPERIMENTS_DIR
    if no_effect: experiments_dir = BOOTSTRAP_EXPERIMENTS_NO_EFFECT_DIR

    os.makedirs(experiments_dir, exist_ok=True)
    src = os.path.join(BOOTSTRAP_DIR, BOOTSTRAP_FILE)
    with open(src, "rb") as f:
        (
            _, _,
            SUBJECTS,
            SUBJECTS_PERMUTATIONS,
            SWITCH,
        ) = pickle.load(f)

    num_draws_processed = 0
    draws_not_processed = []
    arr, reject, correct_reject = [], [], []

    for draw in draws_space:
        curr_arr, curr_reject, curr_correct_reject = [], [], []

        try:
            for M in models:
                for n_subjects in n_subjects_space:
                    match M.NAME:
                        case HierarchicalBayesianModel.NAME:
                            src = os.path.join(
                                experiments_dir,
                                f"d{draw}",
                                f"n{n_subjects}",
                                M.NAME
                            )
                            a_delta_loc = np.load(os.path.join(src, "a_delta_loc.npy"))
                            diff = a_delta_loc[:, 0, :]

                            hdi = hpdi(diff, prob=1 - SIGNIFICANCE_LEVEL, axis=0)
                            decision = (hdi[0, :] > 0) | (hdi[1, :] < 0)
                            curr_reject.append(decision)

                            # Corrected
                            if correction_BonferroniHolm:
                                pr_greater = (diff > 0).mean(axis=0)
                                pr_lesser = (diff < 0).mean(axis=0)
                                pr = np.array([pr_greater, pr_lesser]).max(axis=0)

                                pr_argsort = np.argsort(-pr)
                                pr_inv_argsort = np.argsort(pr_argsort)
                                pr = list(zip(pr, np.arange(pr.shape[0]), pr_inv_argsort))
                                pr = sorted(pr, key=lambda x: x[-1])

                                decision = [False] * len(pr)
                                for value, original_ind, sort_ind in pr:
                                    if value > 1 - (SIGNIFICANCE_LEVEL / (2 * (len(pr) - sort_ind))):
                                        decision[original_ind] = True
                                    else: break

                            else:
                                hdi = hpdi(diff, prob=1 - (SIGNIFICANCE_LEVEL / diff.shape[-1]), axis=0)
                                decision = (hdi[0, :] > 0) | (hdi[1, :] < 0)

                            curr_correct_reject.append(decision)
                            curr_arr.append(decision)

                        case DefaultHierarchicalBayesianModel.NAME:
                            src = os.path.join(
                                experiments_dir,
                                f"d{draw}",
                                f"n{n_subjects}",
                                M.NAME
                            )
                            a = np.load(os.path.join(src, "a_pred.npy"))
                            a = a.mean(axis=0)
                            x = a[:, 0, :]
                            y = a[:, 1, :]
                            assert x.shape == y.shape
                            assert x.shape[0] == n_subjects

                            pr = stats.wilcoxon(
                                x=x - y, alternative="two-sided", axis=0
                            ).pvalue
                            decision = pr < SIGNIFICANCE_LEVEL
                            curr_reject.append(decision)

                            # Corrected
                            if correction_BonferroniHolm:
                                pr_argsort = np.argsort(pr)
                                pr_inv_argsort = np.argsort(pr_argsort)
                                pr = list(zip(pr, np.arange(pr.shape[0]), pr_inv_argsort))
                                pr = sorted(pr, key=lambda x: x[-1])
                                decision = [False] * len(pr)
                                for value, original_ind, sort_ind in pr:
                                    if value < SIGNIFICANCE_LEVEL / (len(pr) - sort_ind):
                                        decision[original_ind] = True
                                    else: break

                            else:
                                decision = pr < (SIGNIFICANCE_LEVEL / pr.shape[-1])

                            curr_correct_reject.append(decision)
                            curr_arr.append(decision)

                        case NonHierarchicalBayesianModel.NAME | MaximumLikelihoodModel.NAME | LeastSquares.NAME:
                            src = os.path.join(BUILD_DIR_PARENT, M.NAME, "a_pred.npy")
                            a = np.load(src)

                            if M.NAME in [NonHierarchicalBayesianModel.NAME, MaximumLikelihoodModel.NAME]:
                                a = a.mean(axis=0)

                            elif M.NAME in [LeastSquares.NAME]:
                                pass

                            else:
                                raise ValueError(f"Unknown model: {M.NAME}")

                            subjects = SUBJECTS_PERMUTATIONS[draw, :n_subjects]
                            subjects = [SUBJECTS[i] for i in subjects]
                            left = [(s, 0) for s in subjects]
                            right = [(s, 1) for s in subjects]

                            if no_effect:
                                switch = SWITCH[draw, :n_subjects]
                                left = [(s, 1) if flag else (s, 0) for s, flag in zip(subjects, switch)]
                                right = [(s, 0) if flag else (s, 1) for s, flag in zip(subjects, switch)]

                            x = np.array([a[*c, :] for c in left])
                            y = np.array([a[*c, :] for c in right])

                            assert np.isnan(x).sum() == 0
                            assert np.isnan(y).sum() == 0
                            assert x.shape == y.shape
                            assert x.shape[0] == n_subjects

                            pr = stats.wilcoxon(
                                x=x - y, alternative="two-sided", axis=0
                            ).pvalue
                            decision = pr < SIGNIFICANCE_LEVEL
                            curr_reject.append(decision)

                            # Corrected
                            if correction_BonferroniHolm:
                                pr_argsort = np.argsort(pr)
                                pr_inv_argsort = np.argsort(pr_argsort)
                                pr = list(zip(pr, np.arange(pr.shape[0]), pr_inv_argsort))
                                pr = sorted(pr, key=lambda x: x[-1])
                                decision = [False] * len(pr)
                                for value, original_ind, sort_ind in pr:
                                    if value < SIGNIFICANCE_LEVEL / (len(pr) - sort_ind):
                                        decision[original_ind] = True
                                    else: break

                            else:
                                decision = pr < (SIGNIFICANCE_LEVEL / pr.shape[-1])

                            curr_correct_reject.append(decision)
                            curr_arr.append(decision)

                        case _:
                            raise ValueError(f"Unknown model: {M.NAME}")

        except FileNotFoundError:
            draws_not_processed.append(draw)
            logger.info(f"Draw: {draw} - Missing {src}")

        else:
            logger.info(f"Draw: {draw}")
            arr += curr_arr
            reject += curr_reject
            correct_reject += curr_correct_reject
            num_draws_processed += 1

    arr = np.array(arr)
    arr = arr.reshape(num_draws_processed, len(models), len(n_subjects_space), *arr.shape[1:])
    logger.info(f"arr.shape: {arr.shape}")

    reject = np.array(reject)
    reject = reject.reshape(num_draws_processed, len(models), len(n_subjects_space), *reject.shape[1:])
    logger.info(f"reject.shape: {reject.shape}")

    correct_reject = np.array(correct_reject)
    correct_reject = correct_reject.reshape(num_draws_processed, len(models), len(n_subjects_space), *correct_reject.shape[1:])
    logger.info(f"correct_reject.: {correct_reject.shape}")

    if no_effect:
        logger.info(arr.any(axis=-1).mean(axis=0))
        logger.info(arr.any(axis=-1).mean(axis=0) - stats.sem(arr.any(axis=-1), axis=0))
        logger.info(arr.any(axis=-1).mean(axis=0) - 1.96 * stats.sem(arr.any(axis=-1), axis=0))

    else:
        logger.info(arr.mean(axis=0))

    ## Debug missing draws
    # res = ""
    # for draw in draws_not_processed: res += f' -e "d{draw}"'
    # res = "grep -rnwl" + res
    # print(" ".join(map(str, draws_not_processed)))
    model_names = [M.NAME for M in models]
    dest = os.path.join(experiments_dir, "results.pkl")
    with open(dest, "wb") as f:
        pickle.dump(
            (
                arr,
                reject,
                correct_reject,
                model_names,
                n_subjects_space,
            ),
            f
        )
    logger.info(f"Saved to: {dest}")
    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )

    draws_space = range(2000)
    n_subjects_space = N_SUBJECTS_SPACE

    models = [
        LeastSquares,
        MaximumLikelihoodModel,
        NonHierarchicalBayesianModel,
        DefaultHierarchicalBayesianModel,
        HierarchicalBayesianModel,
    ]
    no_effect = True

    main(
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        models=models,
        no_effect=no_effect,
        correction_BonferroniHolm=True
    )
