import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/experiments/tms.toml"

TOTAL_SUBJECTS = 8
MAX_INTENSITY = 70
TOTAL_PULSES = 48
TOTAL_REPS = 1

REP = "rep"
INFERENCE_FILE = "inference.pkl"
SIMULATION_DF =  "simulation_df.csv"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-saturation"
LEARN_POSTERIOR_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms-threshold/learn_posterior"
SIMULATE_DATA_DIR = os.path.join(BUILD_DIR, "simulate_data")

EXPERIMENTS_DIR = os.path.join(SIMULATE_DATA_DIR, "experiments")
