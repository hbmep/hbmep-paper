import os
import pickle
import logging

import numpy as np
import pandas as pd
import arviz as az

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import RectifiedLogistic
from constants import (
    TOML_PATH,
    INFERENCE_FILE,
)

logger = logging.getLogger(__name__)


def main():
    # Build model
    M = RectifiedLogistic
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = os.path.join(
        config.BUILD_DIR,
        M.NAME
    )
    model = M(config=config)

    # Set up logging
    model._make_dir(model.build_dir)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    # Load data
    # src = os.path.join(model.csv_path, "sample_data.csv")
    src = model.csv_path
    df = pd.read_csv(src)

    # Run inference
    df, encoder_dict = model.load(df=df)
    logger.info(f"Running inference for {model.NAME} with {df.shape[0]} samples ...")
    mcmc, posterior_samples = model.run_inference(df=df)

    # Predictions and recruitment curves
    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(
        df=prediction_df, posterior_samples=posterior_samples
    )
    model.render_recruitment_curves(
        df=df,
        encoder_dict=encoder_dict,
        posterior_samples=posterior_samples,
        prediction_df=prediction_df,
        posterior_predictive=posterior_predictive
    )
    model.render_predictive_check(
        df=df,
        encoder_dict=encoder_dict,
        prediction_df=prediction_df,
        posterior_predictive=posterior_predictive
    )

    # Save posterior
    dest = os.path.join(model.build_dir, INFERENCE_FILE)
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples,), f)
    logger.info(f"Saved inference data to {dest}")

    # Model evaluation
    inference_data = az.from_numpyro(mcmc)
    logger.info("Evaluating model ...")
    logger.info("LOO ...")
    score = az.loo(inference_data)
    logger.info(score)
    logger.info("WAIC ...")
    score = az.waic(inference_data)
    logger.info(score)
    vars_to_exclude = [site.mu, site.alpha, site.beta, site.obs]
    vars_to_exclude = ["~" + var for var in vars_to_exclude]
    logger.info(az.summary(inference_data, var_names=vars_to_exclude).to_string())


if __name__ == "__main__":
    main()
