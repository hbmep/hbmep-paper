import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/multiple-comparisons/config.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2024-06-08.csv"

BUILD_DIR_PARENT = "/home/vishu/repos/hbmep-paper/reports/multiple-comparisons"
LEARN_POSTERIOR_DIR = os.path.join(BUILD_DIR_PARENT, "learn_posterior")

BUILD_DIR = os.path.join(BUILD_DIR_PARENT, "paired")

TOTAL_SUBJECTS = 32
N_SUBJECTS_SPACE = [2, 4, 6, 8, 12]

TOTAL_PULSES = 48
TOTAL_REPS = 1

REP = "rep"
INFERENCE_FILE = "inference.pkl"
SIMULATION_DF =  "simulation_df.csv"
