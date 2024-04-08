import os


PAPER_DIR = "C:\\Users\\TDT\\Local\\gitprojects\\hbmep-paper"

TOML_PATH = os.path.join(PAPER_DIR, "configs", "basic-setup", "config.toml")
DATA_PATH = os.path.join(PAPER_DIR, "notebooks", "basic-setup", "sample_data.csv")

BUILD_DIR = os.path.join(PAPER_DIR, "reports", "basic-setup")

INFERENCE_FILE = "inference.pkl"
NETCODE_FILE = "inference_data.nc"
