import os
import glob
import pickle
import logging

import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.io import loadmat
from numpyro.diagnostics import summary
from joblib import Parallel, delayed

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import RectifiedLogistic
from utils import read_data
from iterative import DATA_DIR, BUILD_DIR
from summary import INFERENCE_FILE
from utils import MUSCLES


logger = logging.getLogger(__name__)
N_DRAWS = 24
NUM_POINTS = 200
THRESHOLD = 50 * 1e-3
N_DRAWS_TO_PLOT = 5


def plot_foot_draws(model, prediction_df, obs):
    nrows, ncols = N_DRAWS_TO_PLOT, 6
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            model.subplot_cell_width * ncols,
            model.subplot_cell_height * nrows
        ),
        constrained_layout=True,
        squeeze=False
    )
    for draw in range(N_DRAWS_TO_PLOT):
        for r, response in enumerate(model.response):
            ax = axes[draw, r]
            # sns.scatterplot(x=df[model.intensity], y=df[response], ax=ax, color=model.response_colors[r])
            # sns.lineplot(x=prediction_df[model.intensity], y=mu[..., 0, r].mean(axis=0), ax=ax, color="b")
            sns.scatterplot(x=prediction_df[model.intensity], y=obs[draw, ..., r].reshape(-1), ax=ax, color="r", alpha=.2)
    dest = os.path.join(model.build_dir, "five-out-of-ten.png")
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")
    return


def plot_step_function(model, prediction_df, foot):
    nrows, ncols = N_DRAWS_TO_PLOT, 6
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            model.subplot_cell_width * ncols,
            model.subplot_cell_height * nrows
        ),
        constrained_layout=True,
        squeeze=False
    )
    for draw in range(N_DRAWS_TO_PLOT):
        for r, response in enumerate(model.response):
            ax = axes[draw, r]
            sns.lineplot(x=prediction_df[model.intensity].unique(), y=foot[draw, :, r], ax=ax, color=model.response_colors[r])
    dest = os.path.join(model.build_dir, "step-function.png")
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")
    return


def main(visit, participant):
    dir = os.path.join(BUILD_DIR, visit, participant)
    src = os.path.join(dir, INFERENCE_FILE)
    try:
        with open(src, "rb") as f:

            model, mcmc, posterior_samples = pickle.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found: {src}")
        return

    # Setup logging
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__),
        # level=logging.DEBUG
    )

    # Read data
    subdir = os.path.join(DATA_DIR, visit, participant)
    df, mat = read_data(subdir)
    # Remove NaNs
    ind =  df[model.response].isna().any(axis=1)
    df = df[~ind].reset_index(drop=True).copy()
    mat = mat[~ind, ...]
    df, encoder_dict = model.load(df)
    if df.shape[0] < 10: return

    # Make predictions
    prediction_df = model.make_prediction_dataset(df=df, num=NUM_POINTS)
    prediction_df = (
        pd.concat([prediction_df] * N_DRAWS, axis=0)
        .reset_index(drop=True)
        .copy()
    )
    prediction_df = (
        prediction_df
        .sort_values(by=[model.features[0], model.intensity])
        .reset_index(drop=True).copy()
    )
    logger.debug(prediction_df.columns)
    logger.debug(prediction_df.head().to_string())
    posterior_predictive = model.predict(
        df=prediction_df,
        posterior_samples=posterior_samples
    )

    obs = posterior_predictive[site.obs]
    obs = obs.reshape(obs.shape[0], -1, N_DRAWS, obs.shape[-1])
    logger.info(obs.shape)

    mu = posterior_predictive[site.mu]
    mu = mu.reshape(mu.shape[0], -1, N_DRAWS, mu.shape[-1])
    logger.info(mu.shape)

    flag = (obs[..., 0, :] == obs[..., 1, :]).all()
    logger.info(f"Flag obs: {flag}")
    flag = (mu[..., 0, :] == mu[..., 1, :]).all()
    logger.info(f"Flag mu: {flag}")

    foot = (obs > THRESHOLD).mean(axis=-2)
    logger.info(f"foot {foot.shape}")

    # plot_foot_draws(model, prediction_df, obs)
    # plot_step_function(model, prediction_df, foot)
    # mask = (foot < .5).all(axis=-2)
    # mask = mask[..., None, :]
    # mask = np.tile(mask, (1, NUM_POINTS, 1))
    # logger.info(mask.shape)

    # foot = np.ma.array(foot, mask=mask)
    # logger.info(type(foot))

    ind = np.argmax(foot >= .5, axis=-2)
    logger.info(f"ind: {ind.shape}")
    logger.info(type(ind))

    unique_intensity = prediction_df[model.intensity].unique()
    post = unique_intensity[ind]
    logger.info(f"post: {post.shape}")
    logger.info(type(post))

    nrows, ncols = 1, 6
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            model.subplot_cell_width * ncols,
            model.subplot_cell_height * nrows
        ),
        constrained_layout=True,
        squeeze=False
    )

    # bounds = [20] * model.n_response
    # bounds[0] = 30
    summary = []
    target_muscle = df["target_muscle"].unique()[0]
    side = target_muscle[0]

    for r, response in enumerate(model.response):
        threshold = posterior_samples[site.a][:, 0, r].mean()
        samples = post[:, r]
        curr_ind = ind[:, r]
        samples = samples[samples > prediction_df[model.intensity].min()]
        samples = samples[samples > df[model.intensity].min()]
        samples = samples[samples > 5]
        # # samples = samples[samples > threshold - 10]
        # # samples = samples[samples > unique_intensity[10]]

        ax = axes[0, r]
        sns.scatterplot(x=df[model.intensity], y=df[response], ax=ax, color=model.response_colors[r])
        if samples.shape[0] > 100:
            sns.kdeplot(samples, ax=ax, color="green", warn_singular=False)
            ax.axvline(samples.mean(), color="k", linestyle="--", alpha=.4)
        ax.set_title(f"{model.response[r]} - Foot Estimate\nNoise Floor ~ {posterior_samples[site.L][:, 0, r].mean():.2f}")

        result = [
            f"a_foot[0, {r}]",
            samples.mean() if samples.shape[0] > 100 else np.nan,
            az.hdi(samples, hdi_prob=.95)[0] if samples.shape[0] > 100 else np.nan,
            az.hdi(samples, hdi_prob=.95)[1] if samples.shape[0] > 100 else np.nan,
            1 if samples.shape[0] > 100 else np.nan,
            response.split("_")[0] + "_" + side + response.split("_")[1],
            target_muscle
        ]
        summary.append(result)
        logger.info(result)

    dest = os.path.join(model.build_dir, "foot-estimate.png")
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")

    summary_df = pd.DataFrame(summary, columns=["parameter", "mean", "hdi_2.5%", "hdi_97.5%", "r_hat", "muscle", "target_muscle"])
    dest = os.path.join(model.build_dir, "foot-estimate.csv")
    summary_df.to_csv(dest, index=False)
    return

    # # Five out of Ten
    # unique_intensity = prediction_df[model.intensity].unique().sorted()
    # for r, response in enumerate(model.response):
    #     post = []
    #     for intensity in unique_intensity:
    #         ind = prediction_df[model.intensity] == intensity
    #         assert ind.sum() == N_DRAWS
    #         observed_response = obs[]

    # logger.info(prediction_df.head().to_string())
    # return


if __name__ == "__main__":
    src = os.path.join(DATA_DIR, "*")
    visits = glob.glob(src)
    visits = [os.path.basename(v) for v in visits]

    d = {}
    for visit in visits:
        src = os.path.join(DATA_DIR, visit, "*")
        participants = glob.glob(src)
        participants = [os.path.basename(p) for p in participants]
        d[visit] = participants

    subset = [(visit, participant) for visit in visits for participant in d[visit]]
    print(subset)

    # subset = [
    #     ("visit1", "SCS08")
    # ]

    # # Run a single job
    # visit, participant = subset[0]
    # main(visit, participant)

    # Run multiple jobs
    n_jobs = -1
    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(main)(
                visit, participant
            ) \
            for visit, participant in subset
        )
