import os

import numpy as np
from jax import random
import numpyro.distributions as dist

from constants import (
    BUILD_DIR,
    NUM_DRAWS,
    TOTAL_SUBJECTS,
    MEASUREMENTS_PER_SUBJECT,
)

N = 32
N_DRAWS = 2000


def simulate_data(key):
    build_dir = os.path.join(BUILD_DIR, key)
    os.makedirs(build_dir, exist_ok=True)
    rng_key = random.PRNGKey(0)

    n_muscles = 1
    a_scale = 2.5
    scale = 2

    match key:
        case "weak":
            mu = np.array([0.] * n_muscles)
        case "strong1":
            mu = np.array([6.])
        case "strong2":
            mu = np.array([16., 23., 9., 13.])
        case _:
            raise ValueError(f"Invalid key: {key}")

    a_loc_fixed = np.array([45]).reshape(1, -1)
    a_loc_delta = mu.reshape(1, -1)
    a_loc = np.concatenate([a_loc_fixed, a_loc_fixed + a_loc_delta], axis=0)

    a = dist.Normal(a_loc, a_scale).sample(rng_key, (NUM_DRAWS, TOTAL_SUBJECTS,))
    a = np.array(a)

    scale = 2
    y = dist.Normal(a, scale).sample(rng_key, (MEASUREMENTS_PER_SUBJECT,))
    y = np.array(y)

    return y, a, build_dir


# def main():
#     y, a, a_loc, a_scale, scale = simulate_data("strong1")
#     return


# if __name__ == "__main__":
#     main()
