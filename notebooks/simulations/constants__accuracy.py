import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/simulations/accuracy.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2024-06-08.csv"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/simulations"
LEARN_POSTERIOR_DIR = os.path.join(BUILD_DIR, "learn_posterior")

SIMULATE_DATA_DIR__ACCURACY = os.path.join(BUILD_DIR, "accuracy")

EXPERIMENTS_DIR__ACCURACY = os.path.join(SIMULATE_DATA_DIR__ACCURACY, "experiments")
NUMBER_OF_SUBJECTS_DIR = os.path.join(EXPERIMENTS_DIR__ACCURACY, "number_of_subjects")
NUMBER_OF_PULSES_DIR = os.path.join(EXPERIMENTS_DIR__ACCURACY, "number_of_pulses")
NUMBER_OF_REPS_PER_PULSE_DIR = os.path.join(EXPERIMENTS_DIR__ACCURACY, "number_of_reps_per_pulse")

TOTAL_SUBJECTS = 32
N_SUBJECTS_SPACE = [1, 2, 4, 8, 16]

TOTAL_PULSES = 64
N_PULSES_SPACE = [16, 24, 32, 40, 48, 56, 64]

TOTAL_REPS = 8
N_REPS_PER_PULSE_SPACE = [1, 4, 8]

REP = "rep"
INFERENCE_FILE = "inference.pkl"
SIMULATION_DF =  "simulation_df.csv"
