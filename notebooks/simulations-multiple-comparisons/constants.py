import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/simulations-multiple-comparisons/config.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2024-06-08.csv"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/simulations-multiple-comparisons"
LEARN_POSTERIOR_DIR = os.path.join(BUILD_DIR, "learn_posterior")

# SIMULATE_DATA_DIR__ACCURACY = os.path.join(BUILD_DIR, "accuracy")

# EXPERIMENTS_DIR__ACCURACY = os.path.join(SIMULATE_DATA_DIR__ACCURACY, "experiments")
# NUMBER_OF_SUBJECTS_DIR = os.path.join(EXPERIMENTS_DIR__ACCURACY, "number_of_subjects")
# NUMBER_OF_PULSES_DIR = os.path.join(EXPERIMENTS_DIR__ACCURACY, "number_of_pulses")
# NUMBER_OF_REPS_PER_PULSE_DIR = os.path.join(EXPERIMENTS_DIR__ACCURACY, "number_of_reps_per_pulse")

# TOTAL_PULSES = 64
# N_PULSES_SPACE = [16, 24, 32, 40, 48, 56, 64]

# TOTAL_REPS = 8
# N_REPS_PER_PULSE_SPACE = [1, 4, 8]

REP = "rep"
INFERENCE_FILE = "inference.pkl"
SIMULATION_DF =  "simulation_df.csv"

# TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/simulations/power.toml"

# BUILD_DIR = os.path.join(BUILD_DIR, "multiple_comparisons")
# LEARN_POSTERIOR_DIR = os.path.join(BUILD_DIR, "learn_posterior")

# SIMULATE_DATA_DIR__POWER = os.path.join(BUILD_DIR, "power")
# SIMULATE_DATA_WITH_EFFECT_DIR = os.path.join(SIMULATE_DATA_DIR__POWER, "with_effect")
# SIMULATE_DATA_WITH_NO_EFFECT_DIR = os.path.join(SIMULATE_DATA_DIR__POWER, "with_no_effect")

# EXPERIMENTS_WITH_EFFECT_DIR = os.path.join(SIMULATE_DATA_WITH_EFFECT_DIR, "experiments")
# EXPERIMENTS_WITH_NO_EFFECT_DIR = os.path.join(SIMULATE_DATA_WITH_NO_EFFECT_DIR, "experiments")

TOTAL_SUBJECTS = 32
N_SUBJECTS_SPACE = [1, 2, 4, 8, 12, 16, 20]

TOTAL_PULSES = 48
TOTAL_REPS = 1


# def _generate_simulation_data_dirs():
#     return {
#         "with_no_effect": (
#             np.array([0] * 4),
#             np.array([2.5] * 4),
#             os.path.join(BUILD_DIR, "with_no_effect")
#         ),
#         "with_effect": (
#             [5] * 4,
#             [2.5] * 4,
#             os.path.join(BUILD_DIR, "with_effect")
#         ),
#     }
