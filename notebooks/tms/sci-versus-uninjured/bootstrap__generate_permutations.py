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
    NUM_PERMUTATIONS,
    BOOTSTRAP_FILE
)

logger = logging.getLogger(__name__)

BUILD_DIR = BOOTSTRAP_DIR


def main():
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
    df, encoder_dict = model.load(df=df)

    temp = (
        df[model.features]
        .apply(
            lambda x: (
                encoder_dict[model.features[0]].inverse_transform(np.array([x[0]])).item(),
                encoder_dict[model.features[1]].inverse_transform(np.array([x[1]])).item()
            ),
            axis=1
        )
    ).unique()
    sci_participants = sorted([v[0] for v in temp if v[1] == "SCI"])
    uninjured_participants = sorted([v[0] for v in temp if v[1] == "Uninjured"])
    assert len(sci_participants) + len(uninjured_participants) == len(temp)

    sci_participants_encoded = encoder_dict[model.features[0]].transform(sci_participants)
    uninjured_participants_encoded = encoder_dict[model.features[0]].transform(uninjured_participants)

    sci = list(zip(sci_participants, sci_participants_encoded))
    uninjured = list(zip(uninjured_participants, uninjured_participants_encoded))

    # Generate N permutations -- SCI
    rng_key = random.PRNGKey(0)
    sci_permutations = []

    for _ in range(NUM_PERMUTATIONS):
        rng_key, _ = random.split(rng_key)
        ind = np.arange(0, len(sci), 1)
        # With replacement
        ind = random.choice(
            rng_key,
            ind,
            shape=(len(sci),),
            replace=True
        )
        ind = np.array(ind)
        perm = [sci[i] for i in ind]
        sci_permutations.append(perm)

    # Generate N permutations -- Uninjured
    uninjured_permutations = []

    for _ in range(NUM_PERMUTATIONS):
        rng_key, _ = random.split(rng_key)
        ind = np.arange(0, len(uninjured), 1)
        # With replacement
        ind = random.choice(
            rng_key,
            ind,
            shape=(len(uninjured),),
            replace=True
        )
        ind = np.array(ind)
        perm = [uninjured[i] for i in ind]
        uninjured_permutations.append(perm)

    dest = os.path.join(model.build_dir, BOOTSTRAP_FILE)
    with open(dest, "wb") as f:
        pickle.dump((df, encoder_dict, sci_permutations, uninjured_permutations), f)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    main()
