import os
import pickle
import logging

import arviz as az
import pandas as pd

from hbmep.config import Config
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import HierarchicalBayesianModel
from constants import (
    DATA_PATH,
    TOML_PATH,
    INFERENCE_FILE,
    BUILD_DIR
)

logger = logging.getLogger(__name__)


def main():
    M = HierarchicalBayesianModel
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = os.path.join(
        BUILD_DIR,
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
    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)

    # Run inference
    mcmc, posterior_samples_ = model.run_inference(df=df)

    # Predictions and recruitment curves
    posterior_samples = posterior_samples_.copy()
    posterior_samples[site.outlier_prob] = 0 * posterior_samples[site.outlier_prob]
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
        pickle.dump((model, mcmc, posterior_samples_,), f)
    logger.info(f"Saved inference data to {dest}")

    # Model evaluation
    inference_data = az.from_numpyro(mcmc)
    logger.info("Evaluating model ...")
    logger.info("LOO ...")
    score = az.loo(inference_data, var_name=site.obs)
    logger.info(score)
    logger.info("WAIC ...")
    score = az.waic(inference_data, var_name=site.obs)
    logger.info(score)
    vars_to_exclude = [site.mu, site.alpha, site.beta, site.obs]
    vars_to_exclude += [site.q, site.bg_scale]
    vars_to_exclude = ["~" + var for var in vars_to_exclude]
    logger.info(az.summary(inference_data, var_names=vars_to_exclude).to_string())

    return


if __name__ == "__main__":
    main()
