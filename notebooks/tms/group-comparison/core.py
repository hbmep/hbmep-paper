import os
import pickle
import logging

import arviz as az
import pandas as pd

from hbmep.config import Config

from hbmep_paper.utils import setup_logging
from models import MixtureModel

logger = logging.getLogger(__name__)

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/tms/config.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"
FEATURES = ["participant", "participant_condition"]
RESPONSE = ['PKPK_APB', 'PKPK_ECR', 'PKPK_FCR']
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/tms/group-comparison"


def run_inference(model):
    # Load data
    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)

    # Run inference
    # model.plot(df=df, encoder_dict=encoder_dict)
    mcmc, posterior_samples = model.run_inference(df=df)

    # Predict and render plots
    prediction_df = model.make_prediction_dataset(df=df)
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
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "numpyro_data.nc")
    az.to_netcdf(numpyro_data, dest)
    logger.info(dest)

    return


def main(Model):
    # Build model
    config = Config(toml_path=TOML_PATH)
    config.FEATURES = FEATURES
    config.RESPONSE = RESPONSE
    config.BUILD_DIR = os.path.join(BUILD_DIR)
    config.MCMC_PARAMS["num_warmup"] = 5000
    config.MCMC_PARAMS["num_samples"] = 1000
    model = Model(config=config)

    # Setup logging
    model._make_dir(config.BUILD_DIR)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    # Run inference
    run_inference(model)
    return


if __name__ == "__main__":
    main(MixtureModel)