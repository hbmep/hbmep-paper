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
    N_SUBJECTS_SPACE,
    BOOTSTRAP_FILE,
)

logger = logging.getLogger(__name__)

BUILD_DIR = BOOTSTRAP_EXPERIMENTS_DIR


def main(
	build_dir,
	draws_space,
	n_subjects_space,
	models,
	n_jobs=-1
):
    src = os.path.join(BOOTSTRAP_DIR, BOOTSTRAP_FILE)
    with open(src, "rb") as f:
        DF, _, SCI_PERMUTATIONS, UNINJURED_PERMUTATIONS = pickle.load(f)


    def run_experiment(
        n_subjects,
        draw,
        M
    ):
        n_subjects_dir = f"n{n_subjects}"
        draw_dir = f"d{draw}"

        # Build model
        config = Config(toml_path=TOML_PATH)
        config.BUILD_DIR = os.path.join(
            build_dir,
            draw_dir,
            n_subjects_dir,
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
        subjects = [p[1] for p in SCI_PERMUTATIONS[draw][:n_subjects]]
        subjects += [p[1] for p in UNINJURED_PERMUTATIONS[draw][:n_subjects]]
        subjects = sorted(subjects)

        df = []
        for i, subject_ind in enumerate(subjects):
            ind = DF[model.features[0]] == subject_ind
            curr_df = DF[ind].reset_index(drop=True).copy()
            curr_df[model.features[0]] = curr_df[model.features[0]].replace({subject_ind: f"{subject_ind}__{i}"})
            df.append(curr_df)

        df = pd.concat(df, ignore_index=True).copy()
        df = df.reset_index(drop=True).copy()

        # Run inference
        df, encoder_dict = model.load(df=df)
        _, posterior_samples = model.run_inference(df=df)

        # Save
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
        a_loc = None
        del config, df, prediction_df, encoder_dict, _
        del model, posterior_samples, posterior_predictive
        del a_loc
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

    main(
        build_dir=BUILD_DIR,
        draws_space=draws_space,
        n_subjects_space=n_subjects_space,
        models=models,
        n_jobs=n_jobs
    )
