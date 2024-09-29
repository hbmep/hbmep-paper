import os
import pickle
import logging

import pandas as pd

from hbmep.config import Config

from hbmep_paper.utils import setup_logging
from models__accuracy import HierarchicalBayesianModel
from constants__accuracy import (
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
    config.MCMC_PARAMS = {
        "num_warmup": 4000,
        "num_samples": 4000,
        "num_chains": 4,
        "thinning": 4,
    }
    model = HierarchicalBayesianModel(config=config)

    # Set up logging
    os.makedirs(model.build_dir, exist_ok=True)
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
    mcmc, posterior_samples = model.run(df=df)

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
    summary_df = model.summary(posterior_samples)
    logger.info(f"Summary:\n{summary_df.to_string()}")
    dest = os.path.join(model.build_dir, "summary.csv")
    summary_df.to_csv(dest)
    logger.info(f"Saved summary to {dest}")
    logger.info(f"Finished running {model.NAME}")
    logger.info(f"Saved results to {model.build_dir}")
    return


if __name__ == "__main__":
    main()
