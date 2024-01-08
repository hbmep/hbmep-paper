import os
import pickle
import logging
import multiprocessing
from pathlib import Path

import pandas as pd
import jax
import arviz as az
import numpyro

from hbmep.config import Config
from hbmep.model.utils import Site as site

from models import LearnPosterior

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()
numpyro.enable_validation()

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def main():
    toml_path = "/home/vishu/repos/hbmep-paper/configs/experiments/subjects.toml"
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "learn-posterior")

    model = LearnPosterior(config=config)
    model._make_dir(model.build_dir)
    dest = os.path.join(model.build_dir, "log.log")
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

    src = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"
    df = pd.read_csv(src)
    df[model.features[1]] = 0
    df, encoder_dict = model.load(df=df)

    mcmc, posterior_samples = model.run_inference(df=df)

    prediction_df = model.make_prediction_dataset(df=df)
    posterior_predictive = model.predict(df=prediction_df, posterior_samples=posterior_samples)
    model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples, prediction_df=prediction_df, posterior_predictive=posterior_predictive)
    model.render_predictive_check(df=df, encoder_dict=encoder_dict, prediction_df=prediction_df, posterior_predictive=posterior_predictive)

    dest = os.path.join(model.build_dir, "inference.pkl")
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "inference.nc")
    az.to_netcdf(mcmc, dest)
    logger.info(dest)

    numpyro_data = az.from_numpyro(mcmc)
    """ Model evaluation """
    logger.info("Evaluating model ...")
    score = az.loo(numpyro_data, var_name=site.obs)
    logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    score = az.waic(numpyro_data, var_name=site.obs)
    logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")


if __name__ == "__main__":
    main()
