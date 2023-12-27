import os
import pickle
import logging

import arviz as az
import pandas as pd

logger = logging.getLogger(__name__)


def run_inference(model):
    src = "/home/vishu/data/hbmep-processed/J_RCML_000/data.csv"
    df = pd.read_csv(src)
    df, encoder_dict = model.load(df=df)
    mcmc, posterior_samples = model.run_inference(df=df)

    _posterior_samples = posterior_samples.copy()
    # _posterior_samples["outlier_prob"] = _posterior_samples["outlier_prob"] * 0
    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=_posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=_posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

    numpyro_data = az.from_numpyro(mcmc)
    """ Model evaluation """
    logger.info("Evaluating model ...")
    score = az.loo(numpyro_data)
    logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    score = az.waic(numpyro_data)
    logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")

    dest = os.path.join(model.build_dir, "inference.pkl")
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "numpyro_data.nc")
    az.to_netcdf(numpyro_data, dest)
    logger.info(dest)
