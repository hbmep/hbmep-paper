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
    subjects = np.arange(0, 10, 1)
    logger.info(subjects)
    # argsort = subjects.argsort().argsort()
    # logger.info(argsort)
    # # logger.info(subjects[argsort])
    # # logger.info(subjects[argsort][argsort[0]])
    # sorted_subjects = [0] * len(subjects)
    # for i, num in enumerate(argsort):
    #     sorted_subjects[num] = subjects[i]
    # logger.info(sorted_subjects)
    rng_key = jax.random.PRNGKey(0)
    subjects_shuffled = np.array(jax.random.permutation(rng_key, subjects))
    logger.info(subjects_shuffled)
    logger.info(type(subjects_shuffled))
    return


if __name__ == "__main__":
    main()