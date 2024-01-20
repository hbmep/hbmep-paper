import os
import pickle
import logging

import pandas as pd
import arviz as az

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model.utils import Site as site

logger = logging.getLogger(__name__)


def run_inference(config: Config, model: BaseModel, df: pd.DataFrame, do_save=True, make_figures=True):
    df, encoder_dict = model.load(df=df)
    mcmc, posterior_samples = model.run_inference(df=df)

    if make_figures:
        try:
            prediction_df = model.make_prediction_dataset(df=df)
            posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
            model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
            model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
        except:
            print('Failed to make posterior predictive figures. Moving on...')

    if do_save:
        """ Save """
        dest = os.path.join(model.build_dir, "inference.pkl")
        with open(dest, "wb") as f:
            pickle.dump((model, mcmc, posterior_samples, df), f)
        logger.info(dest)

        dest = os.path.join(model.build_dir, "inference.nc")
        az.to_netcdf(mcmc, dest)
        logger.info(dest)

        numpyro_data = az.from_numpyro(mcmc)
        """ Model evaluation """
        logger.info("Evaluating model ...")
        score = az.loo(numpyro_data, var_name=site.obs)
        logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
        score = az.waic(numpyro_data, var_name=site.obs)
        logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")

    return model, mcmc, posterior_samples

