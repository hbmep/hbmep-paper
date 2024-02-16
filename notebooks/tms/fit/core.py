import os
import pickle
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging, run_svi
from models import (
    # MixtureModel,
    RectifiedLogistic,
    Logistic5,
    Logistic4,
    ReLU
)

logger = logging.getLogger(__name__)
LEVEL = logging.INFO

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/rats/J_RCML_000.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/J_RCML_000/data.csv"
FEATURES = [["participant", "compound_position"]]
# FEATURES = ["participant", "compound_position"]
RESPONSE = ["LBiceps", "LECR"]
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/tms/fn-comparison/testing"


def run_inference(model):
    # Load data
    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)
    ind = df[model.features[0]].isin([0, 1])
    df = df[ind].reset_index(drop=True).copy()

    # Run inference
    # mcmc, posterior_samples = model.run_inference(df=df)
    # svi_results, posterior_samples = run_svi(df=df, model=model, **svi_kwargs)
    svi_result, posterior_samples = run_svi(df=df, model=model, steps=20000, lr=1e-2)
    # if model.NAME == "rectified_logistic":
    #     logger.info(f"ell: {posterior_samples[site.ell].mean(axis=0)}")

    losses = np.array(svi_result.losses)
    plt.plot(losses)
    dest = os.path.join(model.build_dir, "losses.png")
    plt.savefig(dest)
    logger.info(f"Saved to {dest}")

    # Predict and render plots
    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

    # Model evaluation
    # numpyro_data = az.from_numpyro(mcmc)
    # logger.info("Evaluating model ...")
    # score = az.loo(numpyro_data)
    # logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    # score = az.waic(numpyro_data)
    # logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")

    # # Save posterior
    # dest = os.path.join(model.build_dir, "inference.pkl")
    # with open(dest, "wb") as f:
    #     pickle.dump((model, mcmc, posterior_samples), f)
    # logger.info(dest)

    return


def main(Model):
    # Build model
    config = Config(toml_path=TOML_PATH)
    config.FEATURES = FEATURES
    config.RESPONSE = RESPONSE
    config.BUILD_DIR = os.path.join(BUILD_DIR, Model.NAME)
    config.MCMC_PARAMS["num_warmup"] = 5000
    config.MCMC_PARAMS["num_samples"] = 1000
    model = Model(config=config)

    # Setup logging
    model._make_dir(config.BUILD_DIR)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__),
        level=LEVEL
    )

    # Run inference
    run_inference(model)
    return


if __name__ == "__main__":
    # Run single model
    # Model = ReLU
    # Model = Logistic4
    Model = Logistic5
    Model = RectifiedLogistic
    main(Model)

    # # Run multiple models in parallel
    # n_jobs = -1
    # models = [RectifiedLogistic, Logistic5, Logistic4, ReLU]
    # with Parallel(n_jobs=n_jobs) as parallel:
    #     parallel(delayed(main)(Model) for Model in models)
