import os
from constants__accuracy import (
    BUILD_DIR
)

REPS_DIR  = os.path.join(BUILD_DIR, "reps")
REPS_DIR__ACCURACY = os.path.join(REPS_DIR, "accuracy")
REPS_DIR__POWER = os.path.join(REPS_DIR, "power")

EXPERIMENTS_DIR__ACCURACY = os.path.join(REPS_DIR__ACCURACY, "experiments")
EXPERIMENTS_DIR__POWER = os.path.join(REPS_DIR__POWER, "experiments")

N_SUBJECTS = 8
TOTAL_SUBJECTS = 32
TOTAL_PULSES = 96

N_SUBJECTS_SPACE = [2, 4, 8, 16, 20]
N_PULSES_SPACE = [48, 72, 96, 120, 144]
N_REPS_SPACE = [1, 8, 12]
