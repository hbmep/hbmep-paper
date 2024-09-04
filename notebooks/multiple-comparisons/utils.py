import os

import numpy as np

from constants__paired import BUILD_DIR as BUILD_DIR_PAIRED
from constants__group import BUILD_DIR as BUILD_DIR_GROUP


def generate_paired_simulation_dirs():
    return {
        "with_effect": (
            np.array([6., 4., -7.5, 8.]),
            np.array([2.5, 3., 3.5, 4.]),
            os.path.join(BUILD_DIR_PAIRED, "with_effect")
        ),
        "no_effect": (
            np.array([0., 0., 0., 0.]),
            np.array([2.5, 3., 3.5, 4.]),
            os.path.join(BUILD_DIR_PAIRED, "no_effect")
        )
    }


def generate_group_simulation_dirs():
    return {
        "with_effect": (
            np.array([9., 12., -8., 13.]),
            os.path.join(BUILD_DIR_GROUP, "with_effect")
        ),
        "no_effect": (
            np.array([0., 0., 0., 0.]),
            os.path.join(BUILD_DIR_GROUP, "no_effect")
        )
    }
