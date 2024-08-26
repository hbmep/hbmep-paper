import os

import numpy as np

from constants__paired import BUILD_DIR as BUILD_DIR_PAIRED


def generate_paired_simulation_dirs():
    return {
        "strong_control": (
            np.array([0, 0, -4, 6]),
            np.array([2.5, 4., 2.5, 3.5]),
            os.path.join(BUILD_DIR_PAIRED, "strong_control")
        )
    }
