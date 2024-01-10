import os
import logging

import pandas as pd
import numpy as np

from hbmep.config import Config
from hbmep.model import BaseModel

from models import RectifiedLogistic, NHBM
from utils import run_inference

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Change this to indicate toml path
TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/experiments/bo.toml"

# Change this to indicate data path
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"


def main():
    # Change this to indicate dataset path
    src = DATA_PATH
    df = pd.read_csv(src)

    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "learn_posterior_iterative")
    base_model = BaseModel(config=config)
    base_model._make_dir(base_model.build_dir)
    df, _ = base_model.load(df=df)

    """ Set up logging in build directory """
    dest = os.path.join(base_model.build_dir, "learn_posterior_iterative.log")
    logging.basicConfig(
        format=FORMAT,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {dest}")

    combinations = base_model._make_combinations(df=df, columns=base_model.features)
    times = []
    loo_scores = []
    waic_scores = []
    for c in combinations:
        logger.info(f"Running for combination {c}")

        config = Config(toml_path=toml_path)
        config.BUILD_DIR = os.path.join(config.BUILD_DIR, "learn_posterior_iterative", "__".join(map(str, c)))
        model = NHBM(config=config)

        ind = df[model.features].apply(tuple, axis=1).isin([c])
        temp_df = df[ind].reset_index(drop=True).copy()

        """ Run inference """
        time_taken, loo, waic = run_inference(config, model, temp_df)
        times.append(time_taken)
        loo_scores.append(loo)
        waic_scores.append(waic)

    logger.info(f"Times: {times}")
    logger.info(f"LOO scores: {loo_scores}")
    logger.info(f"WAIC scores: {waic_scores}")

    times = np.array(times)
    loo_scores = np.array(loo_scores)
    waic_scores = np.array(waic_scores)

    dest = os.path.join(base_model.build_dir, "times.npy")
    np.save(dest, times)
    dest = os.path.join(base_model.build_dir, "loo_scores.npy")
    np.save(dest, loo_scores)
    dest = os.path.join(base_model.build_dir, "waic_scores.npy")
    np.save(dest, waic_scores)
    return


if __name__ == "__main__":
    main()
