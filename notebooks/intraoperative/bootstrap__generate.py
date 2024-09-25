import os
import pickle
import logging

import pandas as pd
from jax import random
import numpy as np

from hbmep.config import Config
from hbmep.model import BaseModel

from hbmep_paper.utils import setup_logging
from constants import (
    DATA_PATH,
    TOML_PATH,
    BUILD_DIR,
    BOOTSTRAP_DIR,
    NUM_BOOTSTRAPS,
    BOOTSTRAP_FILE,
    IS_OUTLIER
)

logger = logging.getLogger(__name__)
BUILD_DIR = BOOTSTRAP_DIR


def main():
    # Build base model for utility functions
    M = BaseModel
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = BUILD_DIR
    model = M(config=config)

    # Set up logging
    os.makedirs(model.build_dir, exist_ok=True)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    # Load data
    df = pd.read_csv(DATA_PATH)
    ind = ~df[model.response].isna().values.any(axis=-1)
    df = df[ind].reset_index(drop=True).copy()
    df = df[~df[IS_OUTLIER]].reset_index(drop=True).copy()
    df, encoder_dict = model.load(df=df)
    logger.info(f"DF shape: {df.shape}")

    # Generate bootstrap permutations (with replacement)
    subjects = df[model.features[0]].unique()
    subjects = sorted(subjects)
    rng_key = random.PRNGKey(0)
    subjects_permutations = random.choice(
        rng_key,
        np.arange(len(subjects)),
        shape=(NUM_BOOTSTRAPS, len(subjects),),
        replace=True
    )
    subjects_permutations = np.array(subjects_permutations)

    # Generate null distribution
    rng_key, _ = random.split(rng_key)
    switch = random.choice(
        rng_key,
        np.array([True, False]),
        shape=(NUM_BOOTSTRAPS, len(subjects),),
        replace=True
    )
    switch = np.array(switch)

    dest = os.path.join(model.build_dir, BOOTSTRAP_FILE)
    with open(dest, "wb") as f:
        pickle.dump(
            (
                df,
                encoder_dict,
                subjects,
                subjects_permutations,
                switch,
            ),
            f
        )
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    main()
