import os
import pickle
import logging

import arviz as az
import pandas as pd
from joblib import Parallel, delayed

from hbmep.config import Config
from models import (
    Existing,
    SD,
    PowerSD,
    PowerSDMinusL
)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/tms/config.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"


def run_inference(data_src, model):
    """ Load data """
    df = pd.read_csv(data_src)

    subset = ["SCA01", "SCA02", "SCA03", "SCA04"]
    ind = df["participant"].isin(subset)
    df = df[ind].reset_index(drop=True).copy()

    df, encoder_dict = model.load(df=df)

    """ Run inference """
    mcmc, posterior_samples = model.run_inference(df=df)

    """ Predict and render plots """
    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

    """ Model evaluation """
    numpyro_data = az.from_numpyro(mcmc)
    logger.info("Evaluating model ...")
    score = az.loo(numpyro_data)
    logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    score = az.waic(numpyro_data)
    logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")

    """ Save posterior """
    dest = os.path.join(model.build_dir, "inference.pkl")
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "numpyro_data.nc")
    az.to_netcdf(numpyro_data, dest)
    logger.info(dest)


def main(data_src, toml_path, features, Model):
    """ Build model """
    config = Config(toml_path=toml_path)
    config.FEATURES = features
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "obs_model", Model.NAME)
    config.RESPONSE = ["PKPK_APB"]
    Model._make_dir(None, config.BUILD_DIR)

    dest = os.path.join(config.BUILD_DIR, "inference.log")
    logging.basicConfig(
        format=FORMAT,
        level=logging.INFO,
        handlers=[
            logging.FileHandler(dest, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {dest}")

    config.MCMC_PARAMS["num_warmup"] = 5000
    config.MCMC_PARAMS["num_samples"] = 1000
    model = Model(config=config)
    run_inference(data_src, model)
    return


if __name__ == "__main__":
    data_src = DATA_PATH
    toml_path = TOML_PATH
    features = [["participant", "participant_condition"]]

    """ Run multiple models in parallel """
    models = [Existing, SD, PowerSD, PowerSDMinusL]
    n_jobs = -1
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(main)(
                data_src, toml_path, features, M
            ) for M in models
        )

    """ Run a single model """
    # Model = PowerSDMinusL
    # main(data_src, toml_path, features, Model)
