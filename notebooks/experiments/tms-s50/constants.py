import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/experiments/tms.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"

TOTAL_SUBJECTS = 8

TOTAL_PULSES = 64
N_PULSES_SPACE = [32, 40, 48, 56, 64]

TOTAL_REPS = 1

REP = "rep"
INFERENCE_FILE = "inference.pkl"
SIMULATION_DF =  "simulation_df.csv"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-s50"
LEARN_POSTERIOR_RECTIFIED_LOGISTIC_DIR = os.path.join(BUILD_DIR, "rectified_logistic")
LEARN_POSTERIOR_LOGISTIC5_DIR = os.path.join(BUILD_DIR, "logistic5")
LEARN_POSTERIOR_LOGISTIC4_DIR = os.path.join(BUILD_DIR, "logistic4")

SIMULATE_DATA_RECTIFIED_LOGISTIC_DIR = os.path.join(LEARN_POSTERIOR_RECTIFIED_LOGISTIC_DIR, "simulate_data")
SIMULATE_DATA_LOGISTIC5_DIR = os.path.join(LEARN_POSTERIOR_LOGISTIC5_DIR, "simulate_data")
SIMULATE_DATA_LOGISTIC4_DIR = os.path.join(LEARN_POSTERIOR_LOGISTIC4_DIR, "simulate_data")

EXPERIMENTS_RECTIFIED_LOGISTIC_DIR = os.path.join(SIMULATE_DATA_RECTIFIED_LOGISTIC_DIR, "experiments")
EXPERIMENTS_LOGISTIC5_DIR = os.path.join(SIMULATE_DATA_LOGISTIC5_DIR, "experiments")
EXPERIMENTS_LOGISTIC4_DIR = os.path.join(SIMULATE_DATA_LOGISTIC4_DIR, "experiments")
