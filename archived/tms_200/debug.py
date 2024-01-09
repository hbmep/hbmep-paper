import logging

import numpy as np

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
    subjects = np.array([8, 4, 9, 2, 1])
    logger.info(subjects)
    argsort = subjects.argsort().argsort()
    logger.info(argsort)
    # logger.info(subjects[argsort])
    # logger.info(subjects[argsort][argsort[0]])
    sorted_subjects = [0] * len(subjects)
    for i, num in enumerate(argsort):
        sorted_subjects[num] = subjects[i]
    logger.info(sorted_subjects)
    return


if __name__ == "__main__":
    main()