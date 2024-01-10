import functools
import os
import pickle
import logging

import pandas as pd
import numpy as np

from hbmep.config import Config
from hbmep.model.utils import Site as site

from models import RectifiedLogistic
from learn_posterior import TOML_PATH, DATA_PATH

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Change this to indicate path to inference.pkl from learn_posterior.py
POSTERIOR_PATH = "/home/mcintosh/Local/temp/test_hbmep/hbmep_sim/build/learn_posterior/inference.pkl"


def main():
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = os.path.join(config.BUILD_DIR, "simulate_data")

    simulator = RectifiedLogistic(config=config)
    simulator._make_dir(simulator.build_dir)

    """ Set up logging in build directory """
    dest = os.path.join(simulator.build_dir, "simulate_data.log")
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

    """ Load learnt posterior """
    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        model, mcmc, posterior_samples = pickle.load(g)

    logger.info(
        f"Logging all sites learnt posterior with their shapes ..."
    )   # Useful for freezing priors
    for k, v in posterior_samples.items():
        logger.info(f"{k}: {v.shape}")

    """
    Simulate new participants
    at equi-spaced intensities.
    We need to freeze the priors.
    """
    # Simulate TOTAL_SUBJECTS subjects
    TOTAL_PULSES = 1
    TOTAL_SUBJECTS = 1
    # Create template dataframe for simulation
    simulation_df = \
        pd.DataFrame(np.arange(0, TOTAL_SUBJECTS, 1), columns=[simulator.features[0]]) \
        .merge(
            pd.DataFrame([0, 90], columns=[simulator.intensity]),
            how="cross"
        )
    simulation_df = simulator.make_prediction_dataset(
        df=simulation_df,
        min_intensity=0,
        max_intensity=0,
        num=TOTAL_PULSES
    )
    logger.info(
        f"Simulation (new participants) dataframe: {simulation_df.shape}"
    )

    sites_to_exclude = {
        site.a, site.b, site.v,
        site.L, site.ell, site.H,
        site.c_1, site.c_2,
        site.mu, site.beta,
        site.obs
    }
    posterior_samples = {
        k: v for k, v in posterior_samples.items() if k not in sites_to_exclude
    }

    simulation_df.loc[0, 'TMSInt'] = 0
    simulation_ppd = \
        simulator.predict(df=simulation_df, posterior_samples=posterior_samples)

    vec_ = list(simulation_ppd.keys())
    vec_ = [item for item in vec_ if item != site.obs]
    vec_ = [item for item in vec_ if item != site.mu]
    vec_ = [item for item in vec_ if item != site.beta]
    for k in vec_:
        posterior_samples[k] = simulation_ppd[k]

    for k in posterior_samples.keys():
        posterior_samples[k] = posterior_samples[k][-5:-4, ...]

    posterior_samples[site.c_2] = posterior_samples[site.c_2] * 1
    posterior_samples[site.c_1] = posterior_samples[site.c_1] * 1

    logger.info(f"Simulating new participants ...")
    x, y, var = [], [], []
    for ix in range(0, 100, 1):
        x_ = ix
        simulation_df.loc[0, 'TMSInt'] = x_
        simulation_ppd = \
            simulator.predict(df=simulation_df, posterior_samples=posterior_samples, random_seed=x_)
        y_ = simulation_ppd['obs'][0][0][0]
        x.append(x_)
        y.append(y_)
        var_ = (simulation_ppd[site.mu] / (posterior_samples[site.c_1] + (posterior_samples[site.c_2]/simulation_ppd[site.mu]))).item()
        var.append(var_)

    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.subplot(1,2, 1)
    plt.plot(x, y, '.')
    plt.ylim([0, 1])
    plt.subplot(1, 2, 2)
    sns.lineplot(x=x, y=var)
    plt.show()
    return


if __name__ == "__main__":
    main()
