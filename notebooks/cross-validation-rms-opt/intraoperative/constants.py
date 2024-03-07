TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/cross-validation/intraoperative.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/intraoperative/data.csv"

N_SPLITS = 8
FOLD_COLUMNS = [f"fold_{i}" for i in range(N_SPLITS)]

PARAMS_FILE = "params.npy"
MSE_FILE = "mse.npy"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/cross-validation-rms-opt/intraoperative"
