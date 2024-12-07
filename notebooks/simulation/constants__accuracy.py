import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/simulation/accuracy.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2024-06-08.csv"

SIMULATION_DIR = "/home/vishu/repos/hbmep-paper/reports/simulation"
LEARN_POSTERIOR_DIR = os.path.join(SIMULATION_DIR, "learn_posterior")

BUILD_DIR = os.path.join(SIMULATION_DIR, "accuracy")
NUMBER_OF_SUBJECTS_DIR = os.path.join(BUILD_DIR, "number_of_subjects")
NUMBER_OF_PULSES_DIR = os.path.join(BUILD_DIR, "number_of_pulses")
NUMBER_OF_REPS_PER_PULSE_DIR = os.path.join(BUILD_DIR, "number_of_reps_per_pulse")

TOTAL_SUBJECTS = 32
TOTAL_REPS = 8

N_SUBJECTS_SPACE = [1, 2, 4, 8, 16]
N_PULSES_SPACE = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]

# TOTAL_PULSES = 64
# N_PULSES_SPACE = [16, 24, 32, 40, 48, 56, 64]

# N_REPS_PER_PULSE_SPACE = [1, 4, 8]

REP = "rep"
INFERENCE_FILE = "inference.pkl"
SIMULATION_DF =  "simulation_df.csv"
