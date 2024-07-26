import os

import numpy as np

from constants import BUILD_DIR


def _generate_simulation_data_dirs():
    return {
        "with_no_effect": (
            np.array([0] * 4),
            np.array([2.5] * 4),
            os.path.join(BUILD_DIR, "with_no_effect")
        ),
        "with_effect": (
            np.array([3, 4, 5, 6]),
            np.array([2.5] * 4),
            os.path.join(BUILD_DIR, "with_effect")
        )
    }
