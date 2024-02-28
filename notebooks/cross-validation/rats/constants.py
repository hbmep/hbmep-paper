import os


TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/cross-validation/J_RCML_000.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/J_RCML_000/data.csv"

INFERENCE_FILE = "inference.pkl"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/cross-validation/rats"
ARVIZ_LOO_DIR = os.path.join(BUILD_DIR, "arviz-loo")
SVI_LOO_DIR = os.path.join(BUILD_DIR, "svi-loo")