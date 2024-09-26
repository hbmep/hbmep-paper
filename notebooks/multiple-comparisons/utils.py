import os

import numpy as np

from constants__paired import BUILD_DIR as BUILD_DIR_PAIRED
from constants__group import BUILD_DIR as BUILD_DIR_GROUP


def generate_paired_simulation_dirs():
    return {
        "sc1": (
            np.array([6., 0., -7.5, 0.]),
            np.array([2.5, 3., 3.5, 4.]),
            os.path.join(BUILD_DIR_PAIRED, "sc1")
        ),
        # "with_effect": (
        #     np.array([6., 4., -7.5, 8.]),
        #     np.array([2.5, 3., 3.5, 4.]),
        #     os.path.join(BUILD_DIR_PAIRED, "with_effect")
        # ),
        # "weak": (
        #     np.array([0., 0., 0., 0.]),
        #     np.array([2.5, 3., 3.5, 4.]),
        #     os.path.join(BUILD_DIR_PAIRED, "no_effect")
        # )
    }


def generate_group_simulation_dirs():
    return {
        "sc1": (
            np.array([0., 13., 0., 8.]),
            os.path.join(BUILD_DIR_GROUP, "sc1")
        ),
    }
