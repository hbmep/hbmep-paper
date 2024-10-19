import os
import gc
import logging

import numpy as np
from joblib import Parallel, delayed

from hbmep.utils import timing

from models import *
from constants import (
    EXPERIMENTS
)
from utils import simulate_data

logger = logging.getLogger(__name__)


@timing
def main(
    key,
    M,
    draws_space,
    n_muscles,
    n_subjects_space,
    n_jobs=-1
):
    y_obs, a_true, build_dir = simulate_data(key=key)
    build_dir = os.path.join(build_dir, EXPERIMENTS)
    os.makedirs(build_dir, exist_ok=True)


    # Define experiment
    def run_experiment(
        n_muscles,
        n_subjects,
        draw,
        M
    ):
        # Build model
        model = M()
        model.build_dir = os.path.join(
            build_dir,
            f"m{n_muscles}",
            f"d{draw}",
            f"n{n_subjects}",
            M.NAME
        )

        y = y_obs[:, draw, :n_subjects, ...]
        y = y[..., :n_muscles]

        n_groups = y.shape[-2]
        subject_ind = np.arange(n_subjects * n_groups)
        subject_ind = subject_ind.reshape(n_groups, n_subjects, ).T
        subject_ind = subject_ind[None, ..., None]
        subject_ind = np.broadcast_to(subject_ind, y.shape)

        group_ind = np.arange(n_groups)
        group_ind = group_ind[None, None, :, None]
        group_ind = np.broadcast_to(group_ind, y.shape)

        y = y.reshape(-1, y.shape[-1])
        subject_ind = subject_ind[..., 0].reshape(-1,)
        group_ind = group_ind[..., 0].reshape(-1,)

        # Run inference
        os.makedirs(model.build_dir, exist_ok=True)
        _, posterior_samples = model.run(
            subject_ind=subject_ind, group_ind=group_ind, y_obs=y
        )

        if "HB" in M.NAME:
            a_loc = posterior_samples["a_loc"]
            np.save(os.path.join(model.build_dir, "a_loc.npy"), a_loc)

            if "a_loc_delta" in posterior_samples:
                a_loc_delta = posterior_samples["a_loc_delta"]
                np.save(os.path.join(model.build_dir, "a_loc_delta.npy"), a_loc_delta)

        elif "NON" in M.NAME:
            a = posterior_samples["a"]
            np.save(os.path.join(model.build_dir, "a_pred.npy"), a)

        model, y, a_loc, a_loc_delta, posterior_samples, _ = [None] * 6
        del model, y, a_loc, a_loc_delta, posterior_samples, _
        gc.collect()
        return


    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(n_muscles, n_subjects, draw, M)
            for draw in draws_space
            for n_subjects in n_subjects_space
        )


if __name__ == "__main__":
    draws_space = range(0, 2000)
    n_subjects_space = [2, 4, 6, 8, 9, 10]
    n_muscles = 4

    key = "strong1"

    # M = HB
    M = HB
    n_jobs = -1

    # M = NON
    # n_jobs = 1
    # n_subjects_space = n_subjects_space[-1:]

    main(
        key=key,
        M=M,
        n_muscles=n_muscles,
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        n_jobs=n_jobs
    )
