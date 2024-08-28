import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/intraoperative/config.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/intraoperative/data.csv"

INFERENCE_FILE = "inference.pkl"
NETCODE_FILE = "inference_data.nc"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/intraoperative/midline-lateral"

BOOTSTRAP_DIR = os.path.join(BUILD_DIR, "bootstrap")
BOOTSTRAP_EXPERIMENTS_DIR = os.path.join(BOOTSTRAP_DIR, "experiments")
BOOTSTRAP_EXPERIMENTS_NO_EFFECT_DIR = os.path.join(BOOTSTRAP_DIR, "experiments_no_effect")

NUM_BOOTSTRAPS = 4000
N_SUBJECTS_SPACE = [2, 4, 6, 8, 10, 13]

BOOTSTRAP_FILE = "bootstrap.pkl"
