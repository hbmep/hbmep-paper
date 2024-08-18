import os

import numpy as np

from constants__group import BUILD_DIR


def _generate_simulation_data_dirs():
    return {
        "with_no_effect": (
            np.array([0., 0., 0., 0.]).reshape(1, -1),
            os.path.join(BUILD_DIR, "with_no_effect")
        ),
        "with_effect": (
            np.array([9., 12., 8., 13.]).reshape(1, -1),
            os.path.join(BUILD_DIR, "with_effect")
        )
    }
