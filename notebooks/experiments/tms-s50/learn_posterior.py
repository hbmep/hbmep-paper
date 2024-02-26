import os
import pickle
import logging

import pandas as pd
import arviz as az

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel
from constants import TOML_PATH, DATA_PATH, LEARN_POSTERIOR_DIR, INFERENCE_FILE

logger = logging.getLogger(__name__)

BUILD_DIR = LEARN_POSTERIOR_DIR


def main():
    # Build model
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = BUILD_DIR
    model = HierarchicalBayesianModel(config=config)

    # Set up logging
    model._make_dir(model.build_dir)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

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
    logger.info(az.summary(inference_data).to_string())
    return


if __name__ == "__main__":
    main()
