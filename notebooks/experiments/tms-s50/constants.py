import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/experiments/tms.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"

TOTAL_SUBJECTS = 16

TOTAL_PULSES = 64
N_PULSES_SPACE = [32, 40, 48, 56, 64]

TOTAL_REPS = 1

REP = "rep"
INFERENCE_FILE = "inference.pkl"
SIMULATION_DF =  "simulation_df.csv"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-s50"
LEARN_POSTERIOR_DIR = os.path.join(BUILD_DIR, "learn_posterior")
SIMULATE_DATA_DIR = os.path.join(LEARN_POSTERIOR_DIR, "simulate_data")

EXPERIMENTS_DIR = os.path.join(SIMULATE_DATA_DIR, "experiments")
NUMBER_OF_SUJECTS_DIR = os.path.join(EXPERIMENTS_DIR, "number_of_subjects")
NUMBER_OF_PULSES_DIR = os.path.join(EXPERIMENTS_DIR, "number_of_pulses")
NUMBER_OF_REPS_PER_PULSE_DIR = os.path.join(EXPERIMENTS_DIR, "number_of_reps_per_pulse")
