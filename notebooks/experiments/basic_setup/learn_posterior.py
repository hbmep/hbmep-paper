import os
import logging

import pandas as pd

from hbmep.config import Config

from models import RectifiedLogistic
from utils import run_inference

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Change this to indicate toml path
TOML_PATH = "/home/mcintosh/Local/gitprojects/hbmep-paper/configs/experiments/basic_setup.toml"

# Change this to indicate data path
DATA_PATH = "/home/mcintosh/Local/temp/test_hbmep/hbmep_sim/data/proc_2023-11-28 1.csv"


def main():
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "learn_posterior")

    # Increase burn-in / samples
    # config.MCMC_PARAMS["num_warmup"] = 2000
    # config.MCMC_PARAMS["num_samples"] = 1000

    model = RectifiedLogistic(config=config)
    model._make_dir(model.build_dir)

    """ Set up logging in build directory """
    dest = os.path.join(model.build_dir, "learn_posterior.log")
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

    # Change this to indicate dataset path
    src = DATA_PATH
    df = pd.read_csv(src)

    """ Filter dataset as required """
    logger.info(df["participant_condition"].unique().tolist())
    ind = df["participant_condition"].isin(["Uninjured"])
    df = df[ind].reset_index(drop=True).copy()

    """ Run inference """
    run_inference(config, model, df)
    return


if __name__ == "__main__":
    main()
