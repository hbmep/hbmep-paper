import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/experiments/tms-power-analysis.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"

TOTAL_SUBJECTS = 16
TOTAL_PULSES = 48
TOTAL_REPS = 1

REP = "rep"
INFERENCE_FILE = "inference.pkl"
SIMULATION_DF =  "simulation_df.csv"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-power-analysis"
LEARN_POSTERIOR_DIR = os.path.join(BUILD_DIR, "learn_posterior")
SIMULATE_DATA_DIR = os.path.join(LEARN_POSTERIOR_DIR, "simulate_data")
SIMULATE_DATA_NO_EFFECT_DIR = os.path.join(LEARN_POSTERIOR_DIR, "simulate_data_no_effect")

EXPERIMENTS_DIR = os.path.join(SIMULATE_DATA_DIR, "experiments")
EXPERIMENTS_NO_EFFECT_DIR = os.path.join(SIMULATE_DATA_NO_EFFECT_DIR, "experiments")
