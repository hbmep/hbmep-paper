import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from hbmep.config import Config
from hbmep.model import functional as F
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import ReLU

logger = logging.getLogger(__name__)

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/rats/J_RCML_000.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/J_RCML_000/data.csv"
FEATURES = [["participant", "compound_position"]]
RESPONSE = ["LBiceps"]
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/svi"


def run_inference(model, df, method, encoder_dict=None):
    match method:
        case "mcmc":
            logger.info("Running MCMC")
            mcmc, posterior_samples = model.run_inference(df=df)
        case "svi":
            logger.info("Running SVI")
            svi_result, posterior_samples = model.run_svi(df=df)
        case _:
            raise ValueError(f"Unknown model: {model.NAME}")

    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    return


def main(Model, method):
    # Build model
    config = Config(toml_path=TOML_PATH)
    config.FEATURES = FEATURES
    config.RESPONSE = RESPONSE
    config.BUILD_DIR = os.path.join(BUILD_DIR, Model.NAME, method)
    config.MCMC_PARAMS["num_warmup"] = 5000
    config.MCMC_PARAMS["num_samples"] = 1000
    model = Model(config=config)
    model._make_dir(model.build_dir)
    setup_logging(dir=model.build_dir, fname=os.path.basename(__file__))

    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)
    # subset = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
    # subset = [(0,), (1,)]
    # ind = df[model.features].apply(tuple, axis=1).isin(subset)
    # df = df[ind].reset_index(drop=True).copy()
    logger.info(f"Data shape: {df.shape}")
    run_inference(model, df, method, encoder_dict=encoder_dict)
    return


if __name__ == "__main__":
    # Model = SVIModel
    Model = ReLU
    method = "svi"
    # method = "mcmc"
    main(Model, method)
