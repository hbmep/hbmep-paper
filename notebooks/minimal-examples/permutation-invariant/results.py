import os
import logging

import arviz as az
import numpy as np
from scipy import stats
from numpyro.diagnostics import hpdi

logger = logging.getLogger(__name__)
N_MUSCLES = 4


def main():
    # dir = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/with_no_effect"
    # dir = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/with_effect_individual"

    # dir = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/paper/with_no_effect"
    dir = "/home/vishu/repos/hbmep-paper/reports/minimal-multiple-comparisons-paired/paper/with_effect"

    n_muscles = N_MUSCLES
    draws_space = range(400)
    n_subjects_space = [2, 4, 8, 12, 16, 20]

    models = ["HB", "HBZero"]
    # models = ["HB", "Uniform"]

    prob = []
    for model_name in models:
        for draw in draws_space:
            for n_subjects in n_subjects_space:
                print(f"draw: {draw}, n_subjects: {n_subjects}")
                src = os.path.join(
                    dir,
                    f"m{n_muscles}",
                    f"d{draw}",
                    f"n{n_subjects}",
                    model_name,
                    "y_loc.npy"
                )
                y_loc = np.load(src)
                prob.append(y_loc)

    prob = np.array(prob)
    prob = prob.reshape(len(models), len(draws_space), len(n_subjects_space), *prob.shape[1:])
    print(prob.shape)

    hdi = hpdi(prob, prob=.95, axis=-2)
    print(hdi.shape)

    arr = ~(((hdi[..., 0, :] < 0) & (hdi[..., 1, :] > 0)).all(axis=-1))
    print(arr.shape)

    print("\n\nIndividual FWER")
    print(f"N_MUSCLES: {n_muscles}, draws: {len(draws_space)}")
    print("Mean")
    print(
        arr.mean(axis=1)
    )
    print(
        (arr.mean(axis=1) < .05).all(axis=-1)
    )
    print("Mean - 1.9 * SEM")
    print(
        arr.mean(axis=1)
        - (1.9 * stats.sem(arr, axis=1))
    )
    print(
        ((arr.mean(axis=1)
        - (1.9 * stats.sem(arr, axis=1))) < .05).all(axis=-1)
    )

    diffs = []
    for i in range(n_muscles):
        for j in range(i + 1, n_muscles):
            diff = prob[..., i] - prob[..., j]
            diffs.append(diff)
    diff = np.stack(diffs, axis=-1)

    hdi = hpdi(diff, prob=.95, axis=-2)
    arr = ~(((hdi[..., 0, :] < 0) & (hdi[..., 1, :] > 0)).all(axis=-1))

    print("\n\nGroup FWER")
    print(f"N_MUSCLES: {n_muscles}, draws: {len(draws_space)}")
    print("Mean")
    print(
        arr.mean(axis=1)
    )
    print(
        (arr.mean(axis=1) < .05).all(axis=-1)
    )
    print("Mean - 1.9 * SEM")
    print(
        arr.mean(axis=1)
        - (1.9 * stats.sem(arr, axis=1))
    )
    print(
        ((arr.mean(axis=1)
        - (1.9 * stats.sem(arr, axis=1))) < .05).all(axis=-1)
    )

    print("Done")


if __name__ == "__main__":
    main()
