import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from jax import random
import jax.numpy as jnp
import numpyro
from numpyro.infer import NUTS, MCMC, Predictive, SVI, Trace_ELBO

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model.utils import Site as site

from hbmep_paper.utils import setup_logging
from models import _model, _guide

logger = logging.getLogger(__name__)

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/rats/J_RCML_000.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/J_RCML_000/data.csv"
FEATURES = [["participant", "compound_position"]]
RESPONSE = ["LBiceps"]
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/rats/J_RCML_000/svi"


def main():
    Model = BaseModel
    toml_path = TOML_PATH
    features = FEATURES
    response = RESPONSE
    config = Config(toml_path=toml_path)
    config.FEATURES = features
    config.RESPONSE = response
    config.BUILD_DIR = os.path.join(BUILD_DIR, Model.NAME)
    model = Model(config=config)

    model._make_dir(model.build_dir)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    src = DATA_PATH
    df = pd.read_csv(src)
    subset = [("amap01", "-C5L")]
    ind = df[model._features[0]].apply(tuple, axis=1).isin(subset)
    df = df[ind].reset_index(drop=True).copy()

    df, encoder_dict = model.load(df=df)
    (features, n_features), (intensity, n_data), = model._collect_regressor(df=df)
    response, = model._collect_response(df=df)
    response = response[:, 0]
    logger.info(f"intensity: {intensity.shape}")
    # logger.info(f"intensity: {type(intensity)}")
    logger.info(f"response: {response.shape}")
    # logger.info(f"response: {type(response)}")

    # """ MCMC """
    # sampler = NUTS(_model)
    # mcmc = MCMC(sampler, **model.mcmc_params)
    # mcmc.run(
    #     model.rng_key,
    #     intensity,
    #     response
    # )
    # mcmc.print_summary(prob=.95)

    # posterior_samples = mcmc.get_samples()
    # posterior_samples = {k: np.array(v) for k, v in posterior_samples.items()}
    # mcmc_mu = posterior_samples[site.mu]
    # logger.info(f"mcmc_mu: {mcmc_mu.shape}")

    # fig, axes = plt.subplots(1, 1, figsize=(10, 5), squeeze=False, constrained_layout=True)
    # ax = axes[0, 0]
    # sns.scatterplot(x=df[model.intensity], y=df[model.response[0]], ax=ax)
    # sns.lineplot(x=df[model.intensity], y=mcmc_mu.mean(axis=0), ax=ax)
    # ax.set_title(
    #     f"a:{posterior_samples[site.a].mean(axis=0).item()} \
    #     \nb:{posterior_samples[site.b].mean(axis=0).item()} \
    #     \neps:{posterior_samples['eps'].mean(axis=0).item()}"
    # )
    # dest = os.path.join(model.build_dir, "mcmc.png")
    # fig.savefig(dest)
    # logger.info(f"Saved to {dest}")

    """ SVI """
    optimizer = numpyro.optim.Adam(step_size=5e-20)
    # auto_guide = numpyro.infer.autoguide.AutoNormal(_model)
    auto_guide = _guide

    svi = SVI(_model, auto_guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(
        model.rng_key,
        1,
        intensity,
        response
    )

    params = svi_result.params
    for u, v in params.items():
        logger.info(f"{u}: {v.shape}")
        logger.info(f"{u}: {v}")

    predictive = Predictive(auto_guide, params=params, num_samples=1000)
    posterior_samples = predictive(model.rng_key, intensity)
    logger.info("Posterior samples:")
    for u, v in posterior_samples.items():
        logger.info(f"{u}: {v.shape}")
        # logger.info(f"{u}: {v.mean(axis=0)}")

    predictive = Predictive(_model, posterior_samples, params=params, num_samples=1000)
    samples = predictive(model.rng_key, intensity)
    logger.info("Samples:")
    for u, v in samples.items():
        logger.info(f"{u}: {v.shape}")
        logger.info(f"{u}: {v.mean(axis=0)}")

    svi_mu = samples[site.mu]
    logger.info(f"svi_mu: {svi_mu.shape}")

    fig, axes = plt.subplots(1, 1, figsize=(10, 5), squeeze=False, constrained_layout=True)
    ax = axes[0, 0]
    sns.scatterplot(x=df[model.intensity], y=df[model.response[0]], ax=ax)
    sns.lineplot(x=df[model.intensity], y=svi_mu.mean(axis=0), ax=ax)
    ax.set_title(
        f"a:{posterior_samples[site.a].mean(axis=0).item()} \
        \nb:{posterior_samples[site.b].mean(axis=0).item()} \
        \neps:{posterior_samples['eps'].mean(axis=0).item()}"
    )
    dest = os.path.join(model.build_dir, "svi.png")
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    main()