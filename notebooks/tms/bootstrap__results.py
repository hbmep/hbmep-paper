import os
import pickle
import logging

import numpy as np
from scipy import stats
from numpyro.diagnostics import hpdi

from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from bootstrap__models import (
    HierarchicalBayesianModel,
    # NonHierarchicalBayesianModel,
    # MaximumLikelihoodModel,
    # NelderMeadOptimization,
)
from constants import (
    BUILD_DIR as BUILD_DIR_PARENT,
    INFERENCE_FILE,
    BOOTSTRAP_DIR,
    BOOTSTRAP_EXPERIMENTS_DIR,
    BOOTSTRAP_EXPERIMENTS_NO_EFFECT_DIR,
    BOOTSTRAP_FILE,
    N_SUBJECTS_SPACE,
)

logger = logging.getLogger(__name__)
BUILD_DIR = BOOTSTRAP_DIR
SIGNIFICANCE_LEVEL = .05
DEBUG = False


def main(
	draws_space,
	n_subjects_space,
	models,
    no_effect
):
    experiments_dir = BOOTSTRAP_EXPERIMENTS_DIR
    if no_effect: experiments_dir = BOOTSTRAP_EXPERIMENTS_NO_EFFECT_DIR

    os.makedirs(experiments_dir, exist_ok=True)
    src = os.path.join(BOOTSTRAP_DIR, BOOTSTRAP_FILE)
    with open(src, "rb") as f:
        (
            _,
            _,
            GROUP_0,
            GROUP_0_PERMUTATIONS,
            GROUP_1,
            GROUP_1_PERMUTATIONS,
            SUBJECTS,
            SUBJECTS_PERMUTATIONS,
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
                            diff = np.load(os.path.join(src, "a_loc_delta.npy"))
                            diff = diff[:, 0, :]

                            # decision = (
                            #     ((diff > 0).mean(axis=0) > 1 - (SIGNIFICANCE_LEVEL / 2))
                            #     | ((diff < 0).mean(axis=0) > 1 - (SIGNIFICANCE_LEVEL / 2))
                            # )
                            hdi = hpdi(diff, prob=1 - SIGNIFICANCE_LEVEL, axis=0)
                            decision = (hdi[0, :] > 0) | (hdi[1, :] < 0)
                            curr_reject.append(decision)
                            curr_arr.append(decision)

                            # Corrected
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

                            curr_correct_reject.append(decision)

                        # case NonHierarchicalBayesianModel.NAME | MaximumLikelihoodModel.NAME | NelderMeadOptimization.NAME:
                        #     src = os.path.join(BUILD_DIR_PARENT, M.NAME, INFERENCE_FILE)

                        #     if M.NAME in [NonHierarchicalBayesianModel.NAME, MaximumLikelihoodModel.NAME]:
                        #         with open(src, "rb") as f:
                        #             posterior_samples, = pickle.load(f)

                        #         a = posterior_samples[site.a]
                        #         a = a.mean(axis=0)

                        #     elif M.NAME == NelderMeadOptimization.NAME:
                        #         with open(src, "rb") as f:
                        #             params, = pickle.load(f)

                        #         a = params[site.a]

                        #     else:
                        #         raise ValueError(f"Unknown model: {M.NAME}")

                        #     group_0 = GROUP_0_PERMUTATIONS[draw, :n_subjects]
                        #     group_1 = GROUP_1_PERMUTATIONS[draw, :n_subjects]
                        #     group_0 = [GROUP_0[i] for i in group_0]
                        #     group_1 = [GROUP_1[i] for i in group_1]

                        #     # Null distribution
                        #     if no_effect:
                        #         group_0 = SUBJECTS_PERMUTATIONS[draw, :n_subjects]
                        #         group_1 = SUBJECTS_PERMUTATIONS[draw, -n_subjects:]
                        #         group_0 = [SUBJECTS[i] for i in group_0]
                        #         group_1 = [SUBJECTS[i] for i in group_1]

                        #     x = np.array([a[*c, :] for c in group_0])
                        #     y = np.array([a[*c, :] for c in group_1])

                        #     assert np.isnan(x).sum() == 0
                        #     assert np.isnan(y).sum() == 0
                        #     assert x.shape == y.shape
                        #     assert x.shape[0] == n_subjects

                        #     pr = stats.ranksums(
                        #         x=x, y=y, alternative="two-sided", axis=0
                        #     ).pvalue
                        #     decision = pr < SIGNIFICANCE_LEVEL
                        #     curr_reject.append(decision)

                        #     # # Corrected
                        #     # pr_argsort = np.argsort(pr)
                        #     # pr_inv_argsort = np.argsort(pr_argsort)
                        #     # pr = list(zip(pr, np.arange(pr.shape[0]), pr_inv_argsort))
                        #     # pr = sorted(pr, key=lambda x: x[-1])
                        #     # decision = [False] * len(pr)
                        #     # for value, original_ind, sort_ind in pr:
                        #     #     if value < SIGNIFICANCE_LEVEL / (len(pr) - sort_ind):
                        #     #         decision[original_ind] = True
                        #     #     else: break
                        #     # curr_correct_reject.append(decision)

                        #     # Corrected (weak)
                        #     pr_argsort = np.argsort(pr)
                        #     pr_inv_argsort = np.argsort(pr_argsort)
                        #     pr = list(zip(pr, np.arange(pr.shape[0]), pr_inv_argsort))
                        #     pr = sorted(pr, key=lambda x: x[-1])
                        #     decision = [False] * len(pr)
                        #     for value, original_ind, sort_ind in pr:
                        #         if value < SIGNIFICANCE_LEVEL / (len(pr) - sort_ind):
                        #             decision[original_ind] = True
                        #         else: break
                        #     curr_correct_reject.append(decision)

                        case _:
                            raise ValueError(f"Unknown model: {M.NAME}")

        except FileNotFoundError:
            draws_not_processed.append(draw)
            logger.info(f"Draw: {draw} - Missing")

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
        print(arr.any(axis=-1).mean(axis=0))

    else:
        print(arr.mean(axis=0))
        # print(arr.mean(axis=0) + stats.sem(arr, axis=0))
    return


if __name__ == "__main__":
    setup_logging(
        dir=BUILD_DIR,
        fname=os.path.basename(__file__)
    )

    draws_space = range(0, 500)
    n_subjects_space = N_SUBJECTS_SPACE
    models = [
        # NelderMeadOptimization,
        # MaximumLikelihoodModel,
        # NonHierarchicalBayesianModel,
        HierarchicalBayesianModel,
    ]
    no_effect = False

    main(
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        models=models,
        no_effect=no_effect
    )
