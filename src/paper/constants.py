import os

HOME = os.getenv("HOME")
REPOS = os.path.join(HOME, "repos", "refactor")
CONFIG = os.path.join(REPOS, "hbmep-paper", "configs")
REPORTS = os.path.join(HOME, "reports")
DATA = os.path.join(HOME, "data", "hbmep-processed")

RAT_TOML = os.path.join(CONFIG, "J_RCML.toml")
RAT_DATA_DIR = os.path.join(DATA, "rat", "J_RCML")
RAT_DATA = os.path.join(RAT_DATA_DIR, "data.csv")

TMS_TOML = os.path.join(CONFIG, "TMS.toml")
# TMS_DATA_DIR = os.path.join(DATA, "human", "2025-04-11_cmct_v0p0p1")
TMS_DATA_DIR = os.path.join(DATA, "human", "tms")
TMS_DATA = os.path.join(TMS_DATA_DIR, "proc_2024-06-08.csv")
TMS_MAT = os.path.join(TMS_DATA_DIR, "proc_2024-06-08.npy")
