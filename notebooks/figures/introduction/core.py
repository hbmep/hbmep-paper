import os
import pickle
import logging

import arviz as az
import pandas as pd

from hbmep.config import Config

from hbmep_paper.utils import setup_logging
from models import (
    RectifiedLogistic,
    NelderMeadOptimization,
)
from constants import (
    TOML_PATH,
    DATA_PATH,
    BUILD_DIR,
)

logger = logging.getLogger(__name__)


def run_inference(model):
    # Load data
    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)

    # Run inference
    mcmc, posterior_samples = model.run_inference(df=df)

    # Predict and render plots
    prediction_df = model.make_prediction_dataset(df=df, num_points=5000, min_intensity=0, max_intensity=105)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

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
        pickle.dump((model, mcmc, posterior_samples, posterior_predictive,), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "prediction_df.csv")
    prediction_df.to_csv(dest, index=False)

    dest = os.path.join(model.build_dir, "numpyro_data.nc")
    az.to_netcdf(numpyro_data, dest)
    logger.info(dest)
    return


def nelder_mead_method(model):
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Run inference
    df, encoder_dict = model.load(df=df)
    params = model.run_inference(df=df)
    logger.info(type(params))
    dest = os.path.join(model.build_dir, "params.pkl")
    with open(dest, "wb") as f:
        pickle.dump((params,), f)
    logger.info(f"Saved to {dest}")

    # Predictions and recruitment curves
    prediction_df = model.make_prediction_dataset(df=df, num_points=5000, min_intensity=0, max_intensity=105)
    prediction_df = model.predict(df=prediction_df, params=params)
    model.render_recruitment_curves(
        df=df,
        encoder_dict=encoder_dict,
        params=params,
        prediction_df=prediction_df,
    )
    return


def main(Model):
    # Build model
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = os.path.join(BUILD_DIR, Model.NAME)
    model = Model(config=config)

    # Setup logging
    model._make_dir(config.BUILD_DIR)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    # Run inference
    match model.NAME:
        case "rectified_logistic":
            run_inference(model)
        case "nelder_mead":
            nelder_mead_method(model)
        case _:
            raise ValueError(f"Unknown model")
    return


if __name__ == "__main__":
    main(RectifiedLogistic)
    main(NelderMeadOptimization)
