import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/intraoperative/config.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/intraoperative/data.csv"

INFERENCE_FILE = "inference.pkl"
NETCODE_FILE = "inference_data.nc"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/intraoperative/midline-lateral"

BOOTSTRAP_DIR = os.path.join(BUILD_DIR, "bootstrap")
NUM_PERMUTATIONS = 4000
BOOTSTRAP_FILE = "bootstrap.pkl"
