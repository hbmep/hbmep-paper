import os
import pickle
import logging

import pandas as pd
from jax import random
import numpy as np
from numpyro import distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel

from hbmep_paper.utils import setup_logging
from constants import (
    DATA_PATH,
    TOML_PATH,
    BUILD_DIR,
    BOOTSTRAP_DIR,
    NUM_BOOTSTRAPS,
    BOOTSTRAP_FILE
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
    df, encoder_dict = model.load(df=df)

    # Get subjects and their encodings
    subjects = encoder_dict[model.features[0]].classes_
    subjects_encoded = encoder_dict[model.features[0]].transform(subjects)
    subjects = list(zip(subjects, subjects_encoded))
    logger.info(subjects)

    # Generate N permutations
    rng_key = random.PRNGKey(0)
    permutations = []
    flags = []

    ind = np.arange(0, len(subjects), 1)
    permutations = random.choice(
        rng_key,
        ind,
        shape=(NUM_BOOTSTRAPS, len(subjects),),
        replace=True
    )
    permutations = np.array(permutations)

    rng_key, _ = random.split(rng_key)
    flags = random.bernoulli(rng_key, .5, (NUM_BOOTSTRAPS, len(subjects),))
    flags = np.array(flags)

    dest = os.path.join(model.build_dir, BOOTSTRAP_FILE)
    with open(dest, "wb") as f:
        pickle.dump((df, encoder_dict, subjects, permutations, flags), f)

    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    main()
