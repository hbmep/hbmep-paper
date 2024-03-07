TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/cross-validation/tms.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"

N_SPLITS = 20
FOLD_COLUMNS = [f"fold_{i}" for i in range(N_SPLITS)]

PARAMS_FILE = "params.npy"
MSE_FILE = "mse.npy"

BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/cross-validation-rms-opt/tms"
