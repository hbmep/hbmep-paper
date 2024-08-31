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
    df[model.features[1]] = (
        df[model.features[1]]
        .replace({
            "Uninjured": "NOT_SCI"
        })
    )
    df, encoder_dict = model.load(df=df)

    subjects = (
        df[model.features]
        .apply(tuple, axis=1)
        .unique()
        .tolist()
    )
    subjects = sorted(subjects)

    group_0 = [c for c in subjects if c[1] == 0]
    group_1 = [c for c in subjects if c[1] == 1]

    # Generate N permutations -- Group 0
    rng_key = random.PRNGKey(0)
    group_0_permutations = random.choice(
        rng_key,
        np.arange(len(group_0)),
        shape=(NUM_BOOTSTRAPS, len(group_0),),
        replace=True
    )
    group_0_permutations = np.array(group_0_permutations)

    # Generate N permutations -- Group 1
    rng_key, _ = random.split(rng_key)
    group_1_permutations = random.choice(
        rng_key,
        np.arange(len(group_1)),
        shape=(NUM_BOOTSTRAPS, len(group_1),),
        replace=True
    )
    group_1_permutations = np.array(group_1_permutations)

    # Null distribution
    rng_key, _ = random.split(rng_key)
    subjects_permutations = random.choice(
        rng_key,
        np.arange(len(subjects)),
        shape=(NUM_BOOTSTRAPS, len(subjects),),
        replace=True
    )

    dest = os.path.join(model.build_dir, BOOTSTRAP_FILE)
    with open(dest, "wb") as f:
        pickle.dump(
            (
                df,
                encoder_dict,
                group_0,
                group_0_permutations,
                group_1,
                group_1_permutations,
                subjects,
                subjects_permutations,
            ),
            f
        )
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    main()
