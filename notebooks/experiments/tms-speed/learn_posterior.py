import os
import pickle
import logging

import pandas as pd
import arviz as az

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import ReLU
from constants import (TOML_PATH, DATA_PATH)

logger = logging.getLogger(__name__)
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-speed/learn_posterior"


def run_inference(model, df, encoder_dict=None):
    # Run inference
    mcmc, posterior_samples = model.run_inference(df=df)

    # Predict and plot recruitment curves
    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

    # Save inference data
    dest = os.path.join(model.build_dir, "inference.pkl")
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "inference.nc")
    az.to_netcdf(mcmc, dest)
    logger.info(dest)

    # Model evaluation
    numpyro_data = az.from_numpyro(mcmc)
    logger.info("Evaluating model ...")
    score = az.loo(numpyro_data, var_name=site.obs)
    logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    score = az.waic(numpyro_data, var_name=site.obs)
    logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")
    return


def main():
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = BUILD_DIR
    config.MCMC_PARAMS["num_warmup"] = 2000
    config.MCMC_PARAMS["num_samples"] = 5000

    model = ReLU(config=config)
    model._make_dir(model.build_dir)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    src = DATA_PATH
    df = pd.read_csv(src)
    df, encoder_dict = model.load(df=df)
    run_inference(model=model, df=df, encoder_dict=encoder_dict)
    return


if __name__ == "__main__":
    main()
