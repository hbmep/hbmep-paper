import os

from jax import random
import numpy as np
import numpyro.distributions as dist

from constants import (
    BUILD_DIR,
    NUM_DRAWS,
    TOTAL_SUBJECTS,
    MEASUREMENTS_PER_SUBJECT,
)


def simulate_data(key=None):
    build_dir = os.path.join(BUILD_DIR, key)
    os.makedirs(build_dir, exist_ok=True)
    rng_key = random.PRNGKey(0)

    n_muscles = 4
    a_scale = 2.5
    scale = 2

    match key:
        case "weak":
            a_loc = [0.] * n_muscles

        case "strong1":
            a_loc = [4., 0., 0., 0.]

        # case "strong_1":
        #     a_loc = [0, 0, -5, 0, 0, 10, 0]

        # case "strong_2":
        #     a_loc = [17, 13, 15, 16, 20, 0, 23]

        # case "strong_3":
        #     a_loc = [17, 0, 15, 16, 20, 0, 23]

        # case "strong_4":
        #     a_loc = [17, 0, 15, 16, 40, 0, 50]

        # # case "strong_0":
        # #     a_loc = [0, 0, 8, -8]

        case _:
            raise ValueError(f"Invalid key: {key}")

    a_loc = np.array(a_loc).reshape(1, -1)

    a = dist.Normal(a_loc, a_scale).sample(rng_key, (NUM_DRAWS, TOTAL_SUBJECTS,))
    a = np.array(a)

    scale = 2
    y = dist.Normal(a, scale).sample(rng_key, (MEASUREMENTS_PER_SUBJECT,))
    y = np.array(y)

    return y, a, build_dir


if __name__ == "__main__":
    y, a, build_dir = simulate_data("strong_0")
    print(y, a, build_dir)
