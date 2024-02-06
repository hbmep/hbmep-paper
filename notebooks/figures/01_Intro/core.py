import os
import pickle
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import scipy.stats as stats

from hbmep.config import Config

from hbmep_paper.utils import setup_logging
from models import (
    RectifiedLogistic,
    NelderMeadOptimization,
    BestPest
)

logger = logging.getLogger(__name__)

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/rats/J_RCML_000.toml"
DATA_PATH = "/home/vishu/repos/hbmep-paper/reports/figures/01_Intro/data.csv"
MAT_PATH = "/home/vishu/repos/hbmep-paper/reports/figures/01_Intro/mat.npy"
RESPONSE = ["APB", "ADM"]
FEATURES = ["participant"]
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/figures/01_Intro"
BINARIZE_THRESHOLD = .07


def run_inference(model):
    # Load data
    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)

    # Run inference
    # model.plot(df=df, encoder_dict=encoder_dict)
    mcmc, posterior_samples = model.run_inference(df=df)

    # Predict and render plots
    prediction_df = model.make_prediction_dataset(df=df, num=5000)
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
    df, encoder_dict = model.load(df=df)

    minimize_result = model.run_inference(df=df)
    logger.info(minimize_result)
    dest = os.path.join(model.build_dir, "minimize_result.pkl")
    with open(dest, "wb") as f:
        pickle.dump((minimize_result,), f)
    logger.info(f"Saved to {dest}")

    prediction_df = model.make_prediction_dataset(df=df, num=5000)
    x_pred = prediction_df[model.intensity].values
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 8, nrows * 3),
        squeeze=False,
        constrained_layout=True
    )
    for muscle_ind in [0, 1]:
        x = df[model.intensity]
        y = df[model.response[muscle_ind]]
        ax = axes[0, muscle_ind]
        sns.scatterplot(x=x, y=y, ax=ax)
        res_x = minimize_result[muscle_ind].x
        y_pred = model.fn(x_pred, *res_x)
        sns.lineplot(x=x_pred, y=y_pred, ax=ax)

    dest = os.path.join(model.build_dir, "pest_data.png")
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")
    return


def pest_method(model):
    logger.info("Running pest method ...")
    # Load data
    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df=df)

    # Binarize data
    binarize_threshold = BINARIZE_THRESHOLD
    df[[response + "_unbin" for response in model.response]] = df[model.response]
    for response in model.response:
        df[response] = df[response].apply(lambda x: 1 if x > binarize_threshold else 0)

    minimize_result = model.run_inference(df=df)
    logger.info(minimize_result)
    dest = os.path.join(model.build_dir, "minimize_result.pkl")
    with open(dest, "wb") as f:
        pickle.dump((minimize_result,), f)
    logger.info(f"Saved to {dest}")

    prediction_df = model.make_prediction_dataset(df=df, num=5000)
    x_pred = prediction_df[model.intensity].values
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 8, nrows * 3),
        squeeze=False,
        constrained_layout=True
    )
    for muscle_ind in [0, 1]:
        x = df[model.intensity]
        y = df[model.response[muscle_ind] + "_unbin"]
        y_bin = df[model.response[muscle_ind]]
        ax = axes[0, muscle_ind]
        ax.set_yticks(np.arange(0, 1, .06))
        ax.grid()
        sns.scatterplot(x=x, y=y, ax=ax)
        ax = axes[1, muscle_ind]
        sns.scatterplot(x=x, y=y_bin, ax=ax)
        # ax.set_xticks(np.arange(0, 100, 1))
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
        ax.grid()
        res_x = minimize_result[muscle_ind].x
        y_pred = model.fn(x_pred, *res_x)
        sns.lineplot(x=x_pred, y=y_pred, ax=ax)
        ax.axvline(x=res_x[0], color="red", linestyle="--")
        quantile_84 = stats.norm.ppf(.84, loc=res_x[0], scale=res_x[1])
        ax.axvline(x=quantile_84, color="red", linestyle="--")
        ax.axvline(x=res_x[0] - (quantile_84 - res_x[0]) , color="red", linestyle="--")

    dest = os.path.join(model.build_dir, "pest_data.png")
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")
    return


def main(Model):
    # Build model
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = os.path.join(BUILD_DIR, Model.NAME)
    config.RESPONSE = RESPONSE
    config.FEATURES = FEATURES
    config.MCMC_PARAMS["num_warmup"] = 10000
    config.MCMC_PARAMS["num_samples"] = 10000
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
        case "best_pest":
            pest_method(model)
    return


if __name__ == "__main__":
    Model = RectifiedLogistic
    # Model = NelderMeadOptimization
    # Model = BestPest
    main(Model)
