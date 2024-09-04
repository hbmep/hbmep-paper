import os
import pickle
import logging

import pandas as pd
import arviz as az

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models__paired import LearnPosteriorModel
from constants__paired import (
    TOML_PATH,
    DATA_PATH,
    LEARN_POSTERIOR_DIR,
    INFERENCE_FILE
)

logger = logging.getLogger(__name__)
BUILD_DIR = LEARN_POSTERIOR_DIR


def main():
    # Build model
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = BUILD_DIR
    config.FEATURES = config.FEATURES[:1]
    config.MCMC_PARAMS["num_warmup"] = 4000
    config.MCMC_PARAMS["num_samples"] = 4000
    config.MCMC_PARAMS["thinning"] = 4

    model = LearnPosteriorModel(config=config)

    # Set up logging
    model._make_dir(model.build_dir)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )
    for u, v in model.mcmc_params.items():
        logger.info(f"{u}: {v}")

    # Load data
    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)

    # Run inference
    mcmc, posterior_samples = model.run_inference(df=df)

    # Save inference data
    dest = os.path.join(model.build_dir, INFERENCE_FILE)
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(f"Saved inference data: {dest}")

    # Predict and plot recruitment curves
    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
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

    # Model evaluation
    logger.info("Evaluating model ...")
    inference_data = az.from_numpyro(mcmc)
    logger.info(az.loo(inference_data, var_name=site.obs))
    vars_to_exclude = [site.mu, site.alpha, site.beta, site.obs]
    vars_to_exclude = ["~" + var for var in vars_to_exclude]
    logger.info(az.summary(inference_data, var_names=vars_to_exclude).to_string())
    return


if __name__ == "__main__":
    main()
