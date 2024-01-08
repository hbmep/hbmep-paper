import functools
import os
import pickle
import logging
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp

import arviz as az
import numpyro
import numpyro.distributions as dist

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model import functional as F
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

    """ Load learn posterior """
    src = "/home/vishu/repos/hbmep-paper/reports/experiments/subjects/learn-posterior/inference.pkl"
    with open(src, "rb") as g:
        model, _, posterior_samples_learnt = pickle.load(g)

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

    for k, v in posterior_samples_learnt.items():
        logger.info(f"{k}: {v.shape}")
    logger.info(f"Before removing sites: {', '.join(posterior_samples_learnt.keys())}")

    """ Create template dataframe for simulation """
    N_SUBJECTS_TO_SIMULATE = 20
    simulation_df = \
        pd.DataFrame(np.arange(0, N_SUBJECTS_TO_SIMULATE, 1), columns=[model.features[0]]) \
        .merge(
            pd.DataFrame(np.arange(0, 1, 1), columns=[model.features[1]]),
            how="cross"
        ) \
        .merge(
            pd.DataFrame([0, 90], columns=[model.intensity]),
            how="cross"
        )
    simulation_df = model.make_prediction_dataset(
        df=simulation_df,
        min_intensity=0,
        max_intensity=100,
        num=60
    )
    logger.info(f"Simulation dataframe: {simulation_df.shape}")

    """ Simulate data """
    priors = {
        "global-priors": [
            "b_scale_global_scale",
            "v_scale_global_scale",
            "L_scale_global_scale",
            "ell_scale_global_scale",
            "H_scale_global_scale",
            "c_1_scale_global_scale",
            "c_2_scale_global_scale"
        ],
        "hyper-priors": [
            "b_scale",
            "v_scale",
            "L_scale",
            "ell_scale",
            "H_scale",
            "c_1_scale",
            "c_2_scale"
        ],
        "baseline-priors": [
            "a_fixed_mean",
            "a_fixed_scale"
        ]
    }
    sites_to_use_for_simulation = priors["global-priors"]
    sites_to_use_for_simulation += priors["hyper-priors"]
    sites_to_use_for_simulation += priors["baseline-priors"]
    # sites_to_use_for_simulation += [site.c_1, site.c_2]
    logger.info(f"sites_to_use_for_simulation: {', '.join(sites_to_use_for_simulation)}")
    posterior_samples_learnt = {
        k: v for k, v in posterior_samples_learnt.items() if k in sites_to_use_for_simulation
    }
    logger.info(f"Removed sites\nRemaining sites: {', '.join(posterior_samples_learnt.keys())}")
    simulation_ppd = \
        model.predict(df=simulation_df, posterior_samples=posterior_samples_learnt)

    for k, v in simulation_ppd.items():
        logger.info(f"{k}: {v.shape}")

    """ Filter valid draws """
    # a = simulation_ppd[site.a]
    # b = simulation_ppd[site.b]
    # H = simulation_ppd[site.H]
    # logger.info(f"a: {a.shape}")
    # logger.info(f"b: {b.shape}")
    # logger.info(f"H: {H.shape}")
    # logger.info(f"a non-valid: {((a > 100).sum() / functools.reduce(lambda x, y: x * y, a.shape)) * 100} %")

    # """ Save simulation dataframe and posterior predictive """
    # dest = os.path.join(model.build_dir, "simulation_df.csv")
    # simulation_df.to_csv(dest, index=False)
    # logger.info(f"Saved simulation dataframe to {dest}")

    # dest = os.path.join(model.build_dir, "simulation_ppd.pkl")
    # with open(dest, "wb") as f:
    #     pickle.dump((model, simulation_ppd), f)
    # logger.info(f"Saved simulation posterior predictive to {dest}")

    """ Plot """
    N_DRAWS_TO_PLOT = 10
    temp_ppd = {
        k: v[:N_DRAWS_TO_PLOT, ...] for k, v in simulation_ppd.items()
    }
    assert temp_ppd[site.obs].shape[0] == N_DRAWS_TO_PLOT

    temp_df = simulation_df.copy()
    temp_ppd = {
        k: v.swapaxes(0, -1) for k, v in temp_ppd.items()
    }
    obs = temp_ppd[site.obs]
    logger.info(f"obs: {obs.shape}")
    response = [model.response[0] + f"_{i}" for i in range(N_DRAWS_TO_PLOT)]
    temp_df[response] = obs[0, ...]
    model.plot(
        df=temp_df,
        response=response,
        response_colors = plt.cm.rainbow(np.linspace(0, 1, N_DRAWS_TO_PLOT)),
        destination_path=os.path.join(model.build_dir, "visual_check.pdf")
    )
    # model.render_recruitment_curves(
    #     df=temp_df,
    #     response=response,
    #     response_colors = plt.cm.rainbow(np.linspace(0, 1, N_DRAWS_TO_PLOT)),
    #     prediction_df=temp_df,
    #     posterior_predictive=temp_ppd,
    #     posterior_samples=temp_ppd,
    #     destination_path=os.path.join(model.build_dir, "visual_check.pdf")
    # )


if __name__ == "__main__":
    main()
