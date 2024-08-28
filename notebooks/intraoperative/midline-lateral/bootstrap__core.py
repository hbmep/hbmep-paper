import os
import sys
import gc
import pickle
import logging

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from hbmep.config import Config

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
        DF, _, SUBJECTS, PERMUTATIONS, FLAGS = pickle.load(f)


    def run_experiment(n_subjects, draw, M):
        # Build model
        config = Config(toml_path=TOML_PATH)
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
        permutation = PERMUTATIONS[draw, :n_subjects]
        flag = FLAGS[draw, :n_subjects]

        subjects = [SUBJECTS[p] for p in permutation]
        subjects = [s[1] for s in subjects]
        subjects = sorted(subjects)

        df = []
        for new_subject_name, subject_name in enumerate(subjects):
            ind = DF[model.features[0]] == subject_name
            curr_df = DF[ind].reset_index(drop=True).copy()
            assert curr_df[model.features[0]].nunique() == 1
            curr_df[model.features[0]] = new_subject_name
            if no_effect and flag[new_subject_name]:
                curr_df[model.features[1]] = (
                    curr_df[model.features[1]]
                    .replace({0: 1, 1: 0})
                )
            df.append(curr_df)

        df = pd.concat(df, ignore_index=True).copy()
        df = df.reset_index(drop=True).copy()

        # Run inference
        df, encoder_dict = model.load(df=df)
        _, posterior_samples = model.run_inference(df=df)

        # Save
        a_delta_loc = posterior_samples["a_delta_loc"]
        a_delta_scale = posterior_samples["a_delta_scale"]
        np.save(os.path.join(model.build_dir, "a_delta_loc.npy"), a_delta_loc)
        np.save(os.path.join(model.build_dir, "a_delta_scale.npy"), a_delta_scale)

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
        a_delta_loc, a_delta_scale = None, None
        del config, df, prediction_df, encoder_dict, _
        del model, posterior_samples, posterior_predictive
        del a_delta_loc, a_delta_scale
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

    no_effect = True
    main(
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        models=models,
        no_effect=no_effect,
        n_jobs=n_jobs
    )
