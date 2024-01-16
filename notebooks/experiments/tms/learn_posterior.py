import os
import pickle
import logging

import pandas as pd
import arviz as az

from hbmep.config import Config
from hbmep.model.utils import Site as site

from models import LearnPosterior
from utils import setup_logging

logger = logging.getLogger(__name__)

# Change paths as necessary
TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/experiments/tms.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/learn_posterior"


def run_inference(model, df, encoder_dict=None):
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
    return


def main():
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = BUILD_DIR

    model = LearnPosterior(config=config)
    model._make_dir(model.build_dir)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    src = DATA_PATH
    df = pd.read_csv(src)
    df[model.features[1]] = 0
    df, encoder_dict = model.load(df=df)
    run_inference(model=model, df=df, encoder_dict=encoder_dict)
    return


if __name__ == "__main__":
    main()
