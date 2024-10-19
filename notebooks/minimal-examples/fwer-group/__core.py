import os
import gc
import logging

import numpy as np
from joblib import Parallel, delayed

from hbmep.utils import timing

from models import NHB, HB, HB2
from constants import (
    EXPERIMENTS
)
from utils import simulate_data

logger = logging.getLogger(__name__)


def _get_data(y_true, n_subjects):
    subject_ind = np.arange(n_subjects * y_true.shape[2]).reshape(2, n_subjects).T
    subject_ind = subject_ind[None, ...]
    subject_ind = np.repeat(subject_ind, y_true.shape[0], axis=0)
    subject_ind = subject_ind[..., None]
    subject_ind = np.repeat(subject_ind, y_true.shape[-1], axis=-1)

    group_ind = np.arange(y_true.shape[2])
    group_ind = group_ind[None, ...]
    group_ind = np.repeat(group_ind, y_true.shape[1], axis=0)
    group_ind = group_ind[None, ...]
    group_ind = np.repeat(group_ind, y_true.shape[0], axis=0)
    group_ind = group_ind[..., None]
    group_ind = np.repeat(group_ind, y_true.shape[-1], axis=-1)

    y_true = y_true.reshape(-1, y_true.shape[-1])
    subject_ind = subject_ind.reshape(-1, subject_ind.shape[-1])
    subject_ind = subject_ind[..., 0]
    group_ind = group_ind.reshape(-1, group_ind.shape[-1])
    group_ind = group_ind[..., 0]
    return subject_ind, group_ind, y_true


@timing
def main(
    key,
    models,
    draws_space,
    n_subjects_space,
    n_muscles_space,
    n_jobs=-1
):
    y_obs, _, build_dir = simulate_data(key=key)
    build_dir = os.path.join(build_dir, EXPERIMENTS)
    os.makedirs(build_dir, exist_ok=True)


    # Define experiment
    def run_experiment(
        n_muscles,
        n_subjects,
        draw,
        M,
    ):
        # Build model
        model = M()
        model.build_dir = os.path.join(
            build_dir,
            f"m{n_muscles}",
            f"d{draw}",
            f"n{n_subjects}",
            M.NAME,
        )

        y_true = y_obs[:, draw, :n_subjects, ...]
        y_true = y_true[..., :n_muscles]
        subject_ind, group_ind, y_true = _get_data(y_true=y_true, n_subjects=n_subjects)

        # Run inference
        os.makedirs(model.build_dir, exist_ok=True)
        mcmc, posterior_samples = model.run_inference(subject_ind=subject_ind, group_ind=group_ind, y_obs=y_true)

        # Save inference
        match M.NAME:
            case HB.NAME | HB2.NAME:
                a_loc = posterior_samples["a_loc"]
                np.save(os.path.join(model.build_dir, "a_loc.npy"), a_loc)

                if M.NAME == HB2.NAME:
                    a_loc_delta = posterior_samples["a_loc_delta"]
                    np.save(os.path.join(model.build_dir, "a_loc_delta.npy"), a_loc_delta)

            case NHB.NAME:
                a = posterior_samples["a"]
                np.save(os.path.join(model.build_dir, "a.npy"), a)

        model, y_true, a_loc, a, mcmc, posterior_samples = None, None, None, None, None, None
        del model, y_true, a_loc, a, mcmc, posterior_samples
        gc.collect()
        return


    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(n_muscles, n_subjects, draw, M)
            for n_muscles in n_muscles_space
            for draw in draws_space
            for n_subjects in n_subjects_space
            for M in models
        )


if __name__ == "__main__":
    draws_space = range(200, 1000)
    n_muscles_space = [4]

    # # models = [HB2]
    # models = [HB, HB2]
    models = [HB]
    n_subjects_space = [2, 4, 6, 8, 10, 12]

    # models = [NHB]
    # n_subjects_space = [12]

    key = "strong"
    # key = "weak"
    main(
        key=key,
        models=models,
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        n_muscles_space=n_muscles_space,
        n_jobs=1 if NHB in models else -1,
    )
