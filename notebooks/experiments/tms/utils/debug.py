import logging

import numpy as np
import jax

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
dest = "/home/vishu/debug.log"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(dest, mode="w"),
        logging.StreamHandler()
    ],
    force=True
)


def main():
    total_pulses = 15
    pulses = np.arange(0, total_pulses, 1).astype(int)
    logger.info(pulses)
    logger.info(type(pulses))
    logger.info(pulses.dtype)
    # argsort = subjects.argsort().argsort()
    # logger.info(argsort)
    # # logger.info(subjects[argsort])
    # # logger.info(subjects[argsort][argsort[0]])
    # sorted_subjects = [0] * len(subjects)
    # for i, num in enumerate(argsort):
    #     sorted_subjects[num] = subjects[i]
    # logger.info(sorted_subjects)
    # rng_key = jax.random.PRNGKey(0)
    # subjects_shuffled = np.array(jax.random.permutation(rng_key, subjects))
    # logger.info(subjects_shuffled)
    # logger.info(type(subjects_shuffled))
    return


if __name__ == "__main__":
    main()