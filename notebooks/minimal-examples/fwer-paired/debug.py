import os
import logging

import numpy as np
from scipy import stats
from numpyro.diagnostics import hpdi, gelman_rubin

from models import *
from constants import (
    EXPERIMENTS
)

logger = logging.getLogger(__name__)
np.set_printoptions(suppress=True)

SIGNIFICANCE_LEVEL = .05


def main(experiments_dir, ind, draws_space, n_subjects_space, models):
    power_ind, t1_ind = ind
    n_muscles = len(power_ind) + len(t1_ind)

    num_draws_processed = 0
    draws_not_processed = []
    arr, reject, correct_reject = [], [], []

    for draw in draws_space:
        curr_arr, curr_reject, curr_correct_reject = [], [], []

        try:
            for M in models:
                for n_subjects in n_subjects_space:
                    if "HB" in M.NAME:
                        src = os.path.join(
                            experiments_dir,
                            f"m{n_muscles}",
                            f"d{draw}",
                            f"n{n_subjects}",
                            M.NAME
                        )
                        a_loc = np.load(os.path.join(src, "a_loc.npy"))
                        diff = a_loc

                        if n_subjects == n_subjects_space[-1]:
                            # a_loc_scale = np.load(os.path.join(src, "a_loc_scale.npy"))

                            nr, nc, figsize = 2, 4, (12, 4)
                            # nr, nc, figsize = 4, 4, (12, 6)
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            fig, axes = plt.subplots(nr, nc, squeeze=False, constrained_layout=True, figsize=figsize, sharex="row", sharey="row")
                            for j in range(diff.shape[-1]):
                                ax = axes[1, j]
                                samples = diff[:, j]
                                sns.kdeplot(samples, ax=axes[0, j])
                                samples = samples.reshape(4, -1)
                                for i in range(4):
                                    sns.lineplot(x=np.arange(samples.shape[-1]), y=samples[i, :], ax=ax)

                                ax.axhline(0, color="black", linestyle="--", alpha=.5)

                                # if a_loc_scale.ndim == 2: samples = a_loc_scale[:, j]
                                # else: samples = a_loc_scale

                                # sns.kdeplot(samples, ax=axes[2, j])
                                # samples = samples.reshape(4, -1)
                                # for i in range(4):
                                #     sns.lineplot(x=np.arange(samples.shape[-1]), y=samples[i, :], ax=axes[3, j])

                            fig.savefig(os.path.join(src, "diff.png"))
                            plt.close()

                    elif "NON" in M.NAME:
                        src = os.path.join(
                            experiments_dir,
                            f"m{n_muscles}",
                            f"d{draw}",
                            f"n{n_subjects_space[-1]}",
                            M.NAME
                        )
                        a = np.load(os.path.join(src, "a_pred.npy"))
                        a = a.mean(axis=0)

                        x = a[:n_subjects, 0, :]
                        y = a[-n_subjects:, 1, :]
                        assert np.isnan(x).sum() == 0
                        assert np.isnan(y).sum() == 0
                        assert x.shape == y.shape
                        assert x.shape[0] == n_subjects

                        pr = stats.ranksums(
                            x=x, y=y, alternative="two-sided", axis=0
                        ).pvalue
                        # pr = stats.ttest_ind(
                        #     a=x, b=y, alternative="two-sided", axis=0, equal_var=False
                        # ).pvalue
                        decision = pr < SIGNIFICANCE_LEVEL
                        curr_reject.append(decision)

                        # Corrected
                        pr_argsort = np.argsort(pr)
                        pr_inv_argsort = np.argsort(pr_argsort)
                        pr = list(zip(pr, np.arange(pr.shape[0]), pr_inv_argsort))
                        pr = sorted(pr, key=lambda x: x[-1])
                        decision = [False] * len(pr)
                        for value, original_ind, sort_ind in pr:
                            if value < SIGNIFICANCE_LEVEL / (len(pr) - sort_ind):
                                decision[original_ind] = True
                            else: break
                        curr_correct_reject.append(decision)
                        curr_arr.append(decision)

                    else:
                        raise ValueError(f"Unknown model: {M.NAME}")

        except FileNotFoundError:
            draws_not_processed.append(draw)
            print(f"Draw: {draw} - Missing")

        else:
            print(f"Draw: {draw}")
            arr += curr_arr
            reject += curr_reject
            correct_reject += curr_correct_reject
            num_draws_processed += 1

    reject = np.array(reject)
    reject = reject.reshape(num_draws_processed, len(models), len(n_subjects_space), *reject.shape[1:])
    logger.info(f"reject.shape: {reject.shape}")

    correct_reject = np.array(correct_reject)
    correct_reject = correct_reject.reshape(num_draws_processed, len(models), len(n_subjects_space), *correct_reject.shape[1:])
    logger.info(f"correct_reject.: {correct_reject.shape}")

    arr = np.array(arr)
    arr = arr.reshape(num_draws_processed, len(models), len(n_subjects_space), *arr.shape[1:])
    print(f"arr.shape: {arr.shape}")

    print(arr[..., power_ind].mean(axis=0)[..., 0])
    print()
    print(arr[..., t1_ind].any(axis=-1).mean(axis=0))
    # print(arr[..., t1_ind].any(axis=-1).mean(axis=0) - 1.96 * stats.sem(arr[..., t1_ind].any(axis=-1), axis=0))
    return


if __name__ == "__main__":
    draws_space = range(500)
    n_subjects_space = [2, 4, 6, 7, 8, 9, 10, 11, 12]
    n_subjects_space = n_subjects_space[-1:]
    models = [
        # NON,
        HB2
    ]

    key, ind = "strong1", ([0], [1, 2, 3])
    experiments_dir = f"/home/vishu/repos/hbmep-paper/reports/minimal-examples/fwer-paired/{key}/experiments"

    # key, ind = "weak", ([], [0, 1, 2, 3])
    # experiments_dir = f"/home/vishu/repos/hbmep-paper/reports/minimal-examples/fwer-paired/{key}/experiments"
    main(
        experiments_dir=experiments_dir,
        ind=ind,
        draws_space=draws_space,
        models=models,
        n_subjects_space=n_subjects_space
    )
