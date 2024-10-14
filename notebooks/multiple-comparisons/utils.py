import os

import numpy as np

from constants__paired import BUILD_DIR as BUILD_DIR_PAIRED
from constants__group import BUILD_DIR as BUILD_DIR_GROUP


def generate_paired_simulation_dirs():
    return {
        "sc1": (
            np.array([-6., 0., 0., 0.]),
            np.array([2.5, 3., 3.5, 4.]),
            os.path.join(BUILD_DIR_PAIRED, "sc1")
        ),
        "weak": (
            np.array([0., 0., 0., 0.]),
            np.array([2.5, 3., 3.5, 4.]),
            os.path.join(BUILD_DIR_PAIRED, "weak")
        ),
    }


def generate_group_simulation_dirs():
    return {
        "sc1": (
            np.array([18., 0., 0., 0.]),
            os.path.join(BUILD_DIR_GROUP, "sc1")
        ),
    }
