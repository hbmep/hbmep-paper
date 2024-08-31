import os
import sys
import gc
import pickle
import logging

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.utils import timing

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel
from constants import (
    TOML_PATH,
    BOOTSTRAP_DIR,
    BOOTSTRAP_EXPERIMENTS_DIR,
    BOOTSTRAP_EXPERIMENTS_NO_EFFECT_DIR,
    N_SUBJECTS_SPACE,
    BOOTSTRAP_FILE,
)

logger = logging.getLogger(__name__)
BUILD_DIR = BOOTSTRAP_EXPERIMENTS_DIR


@timing
def main(
	draws_space,
	n_subjects_space,
	models,
    no_effect,
	n_jobs=-1
):
    build_dir = BOOTSTRAP_EXPERIMENTS_DIR
    if no_effect: build_dir = BOOTSTRAP_EXPERIMENTS_NO_EFFECT_DIR
    os.makedirs(build_dir, exist_ok=True)

    src = os.path.join(BOOTSTRAP_DIR, BOOTSTRAP_FILE)
    with open(src, "rb") as f:
        (
            DF,
            _,
            GROUP_0,
            GROUP_0_PERMUTATIONS,
            GROUP_1,
            GROUP_1_PERMUTATIONS,
            SUBJECTS,
            SUBJECTS_PERMUTATIONS,
        ) = pickle.load(f)


    def run_experiment(n_subjects, draw, M):
        # Build model
        config = Config(toml_path=TOML_PATH)
        config.MCMC_PARAMS["num_warmup"] = 4000
        config.MCMC_PARAMS["num_samples"] = 1000
        config.MCMC_PARAMS["thinning"] = 1
        config.BUILD_DIR = os.path.join(
            build_dir,
            f"d{draw}",
            f"n{n_subjects}",
            M.NAME
        )
        model = M(config=config)

        # Set up logging
        os.makedirs(model.build_dir, exist_ok=True)
        setup_logging(
            dir=model.build_dir,
            fname="logs"
        )

        # Load data
        group_0 = GROUP_0_PERMUTATIONS[draw, :n_subjects]
        group_1 = GROUP_1_PERMUTATIONS[draw, :n_subjects]
        group_0 = [GROUP_0[i] for i in group_0]
        group_1 = [GROUP_1[i] for i in group_1]

        # Null distribution
        if no_effect:
            group_0 = SUBJECTS_PERMUTATIONS[draw, :n_subjects]
            group_1 = SUBJECTS_PERMUTATIONS[draw, -n_subjects:]
            group_0 = [SUBJECTS[i] for i in group_0]
            group_1 = [SUBJECTS[i] for i in group_1]
            group_0 = [(c[0], 0) for c in group_0]
            group_1 = [(c[0], 1) for c in group_1]

        subjects = group_0 + group_1
        subjects = sorted(subjects, key=lambda x: (x[1], x[0]))

        df = []
        for new_subject_name, (subject_ind, _) in enumerate(subjects):
            ind = DF[model.features[0]] == subject_ind
            curr_df = DF[ind].reset_index(drop=True).copy()
            assert curr_df[model.features[0]].nunique() == 1
            curr_df[model.features[0]] = new_subject_name
            df.append(curr_df)

        df = pd.concat(df, ignore_index=True).copy()
        df = df.reset_index(drop=True).copy()

        # Run inference
        df, encoder_dict = model.load(df=df)
        _, posterior_samples = model.run_inference(df=df)

        # Save
        a_loc_delta = posterior_samples["a_loc_delta"]
        np.save(os.path.join(model.build_dir, "a_loc_delta.npy"), a_loc_delta)
        a_loc = posterior_samples["a_loc"]
        np.save(os.path.join(model.build_dir, "a_loc.npy"), a_loc)

        # Predictions and recruitment curves
        prediction_df = model.make_prediction_dataset(df=df)
        posterior_predictive = model.predict(
            df=prediction_df, posterior_samples=posterior_samples
        )
        model.render_recruitment_curves(
            df=df,
            encoder_dict=encoder_dict,
            posterior_samples=posterior_samples,
            prediction_df=prediction_df,
            posterior_predictive=posterior_predictive
        )

        config, df, prediction_df, encoder_dict, _, = None, None, None, None, None
        model, posterior_samples, posterior_predictive = None, None, None
        a_loc_delta, a_loc, = None, None
        del config, df, prediction_df, encoder_dict, _
        del model, posterior_samples, posterior_predictive
        del a_loc_delta, a_loc
        gc.collect()


    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_experiment)(n_subjects, draw, M)
            for draw in draws_space
            for n_subjects in n_subjects_space
            for M in models
        )

    return


if __name__ == "__main__":
    # Usage: python -m bootstrap__core.py 0 100
    lo, hi = list(map(int, sys.argv[1:]))

    # Experiment space
    draws_space = range(lo, hi)
    n_jobs = -1

    # Run hierarchical models
    n_subjects_space = N_SUBJECTS_SPACE
    models = [
        HierarchicalBayesianModel
    ]

    no_effect = False
    main(
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        models=models,
        no_effect=no_effect,
        n_jobs=n_jobs
    )
