import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/tms/config.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2024-06-08.csv"

INFERENCE_FILE = "inference.pkl"
NETCODE_FILE = "inference_data.nc"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/tms-sci-vs-uninjured"

BOOTSTRAP_DIR = os.path.join(BUILD_DIR, "bootstrap")
BOOTSTRAP_EXPERIMENTS_DIR = os.path.join(BOOTSTRAP_DIR, "experiments")

NUM_BOOTSTRAPS = 4000
N_SUBJECTS_SPACE = [2, 4, 6, 8, 10, 13]

BOOTSTRAP_FILE = "bootstrap.pkl"
