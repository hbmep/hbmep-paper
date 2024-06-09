import os

from constants__accuracy import (
    BUILD_DIR
)

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/simulations/power.toml"

SIMULATE_DATA_DIR__POWER = os.path.join(BUILD_DIR, "power")
SIMULATE_DATA_WITH_EFFECT_DIR = os.path.join(SIMULATE_DATA_DIR__POWER, "with_effect")
SIMULATE_DATA_WITH_NO_EFFECT_DIR = os.path.join(SIMULATE_DATA_DIR__POWER, "with_no_effect")

EXPERIMENTS_WITH_EFFECT_DIR = os.path.join(SIMULATE_DATA_WITH_EFFECT_DIR, "experiments")
EXPERIMENTS_WITH_NO_EFFECT_DIR = os.path.join(SIMULATE_DATA_WITH_NO_EFFECT_DIR, "experiments")

TOTAL_SUBJECTS = 32
N_SUBJECTS_SPACE = [1, 2, 4, 8, 12, 16, 20]

TOTAL_PULSES = 48
TOTAL_REPS = 1
