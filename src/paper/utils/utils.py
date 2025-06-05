import os
import pickle
import logging

import arviz as az
import numpy as np
import numpyro
from numpyro.infer import Predictive, SVI, Trace_ELBO

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATEFMT = "%b-%d-%Y %I:%M:%S %p"


def setup_logging(dir, fname, level=logging.INFO):
    fname = f"{fname.split('.')[0]}.log"
    dest = os.path.join(
        dir, fname
    )
    logging.basicConfig(
        format=FORMAT,
        datefmt=DATEFMT,
        level=level,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {dest}")
    return


def run_inference(df, model, encoder_dict=None):
    mcmc, posterior_samples = model.run_inference(df=df)
    # Predict and render plots
    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    return mcmc, posterior_samples


def evaluate_and_save(model, mcmc, posterior_samples):
    # Model evaluation
    numpyro_data = az.from_numpyro(mcmc)
    logger.info("Evaluating model ...")
    score = az.loo(numpyro_data)
    logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    score = az.waic(numpyro_data)
    logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")

    # Save posterior
    dest = os.path.join(model.build_dir, "inference.pkl")
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "numpyro_data.nc")
    az.to_netcdf(numpyro_data, dest)
    logger.info(dest)

    return


def run_svi(
    df,
    model,
    lr=1e-2,
    steps=2000,
    PROGRESS_BAR=True,
):
    optimizer = numpyro.optim.ClippedAdam(step_size=lr)
    _guide = numpyro.infer.autoguide.AutoLowRankMultivariateNormal(model._model)
    svi = SVI(
        model._model,
        _guide,
        optimizer,
        loss=Trace_ELBO(num_particles=20)
    )
    svi_result = svi.run(
        model.rng_key,
        steps,
        *model._get_regressors(df=df),
        *model._get_response(df=df),
        progress_bar=PROGRESS_BAR
    )
    predictive = Predictive(
        _guide,
        params=svi_result.params,
        num_samples=4000
    )
    posterior_samples = predictive(model.rng_key, *model._get_regressors(df=df))
    posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
    return svi_result, posterior_samples
