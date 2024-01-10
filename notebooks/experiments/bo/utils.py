import os
import pickle
import logging
import time

import pandas as pd
import arviz as az

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model.utils import Site as site

logger = logging.getLogger(__name__)


def run_inference(config: Config, model: BaseModel, df: pd.DataFrame):
    df, encoder_dict = model.load(df=df)

    start = time.time()
    mcmc, posterior_samples = model.run_inference(df=df)
    end = time.time()
    time_taken = end - start
    logger.info(f"Time taken: {time_taken:.2f} seconds")

    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

    """ Save """
    dest = os.path.join(model.build_dir, "inference.pkl")
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "inference.nc")
    az.to_netcdf(mcmc, dest)
    logger.info(dest)

    numpyro_data = az.from_numpyro(mcmc)
    """ Model evaluation """
    logger.info("Evaluating model ...")
    loo = az.loo(numpyro_data, var_name=site.obs).elpd_loo
    logger.info(f"ELPD LOO (Log): {loo:.2f}")
    waic = az.waic(numpyro_data, var_name=site.obs).elpd_waic
    logger.info(f"ELPD WAIC (Log): {waic:.2f}")

    return time_taken, loo, waic
