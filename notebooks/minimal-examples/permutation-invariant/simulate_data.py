import os
import pickle
import logging

import pandas as pd
import numpy as np
from jax import random
import numpyro.distributions as dist

from constants import (
    BUILD_DIR
)

N = 32
N_DRAWS = 2000
N_MUSCLES = 20


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    rng_key = random.PRNGKey(0)

    # With no effect
    mu = np.array([0.] * N_MUSCLES)
    sigma = np.array([2.5] * N_MUSCLES)
    y = dist.Normal(mu, sigma).sample(rng_key, (2000, N,))
    y = np.array(y)
    dest = os.path.join(BUILD_DIR, "with_no_effect.pkl")
    with open(dest, "wb") as f:
        pickle.dump((mu, sigma, y), f)

    # With effect
    mu = np.array([5] * N_MUSCLES)
    sigma = np.array([2.5] * N_MUSCLES)
    y = dist.Normal(mu, sigma).sample(rng_key, (2000, N,))
    y = np.array(y)
    dest = os.path.join(BUILD_DIR, "with_effect_individual.pkl")
    with open(dest, "wb") as f:
        pickle.dump((mu, sigma, y), f)

    return


if __name__ == "__main__":
    main()
