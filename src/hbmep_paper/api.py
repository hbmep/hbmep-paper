import os
import logging
import multiprocessing

import jax
import numpyro

from hbmep.config import Config

from hbmep_paper.model import (
    Simulator,
    HierarchicalBayesian,
    NonHierarchicalBayesian,
    MaximumLikelihood
)
from hbmep_paper.model.non_centered import NonCenteredHierarchicalBayesian
from hbmep_paper.model.cauchy import CauchyModel
from hbmep_paper.utils import simulate, run_experiment

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
numpyro.set_platform(PLATFORM)

cpu_count = multiprocessing.cpu_count() - 2
numpyro.set_host_device_count(cpu_count)
numpyro.enable_x64()

logger = logging.getLogger(__name__)


def run(config: Config):
    simulator = Simulator(config=config)
    simulation_params = {
        "n_subject": 5,
        "n_feature0": 15,
        "n_repeats": 100,
        "downsample_rate": 1
    }
    df, posterior_samples = simulate(model=simulator, **simulation_params)

    models = [NonCenteredHierarchicalBayesian, CauchyModel]

    run_experiment(
        config=config, models=models, df=df, posterior_samples_true=posterior_samples
    )
    return
