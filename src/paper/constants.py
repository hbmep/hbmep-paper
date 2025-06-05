import os

HOME = os.getenv("HOME")
REPOS = os.path.join(HOME, "repos", "refactor")
CONFIG = os.path.join(REPOS, "hbmep-paper", "configs")
REPORTS = os.path.join(HOME, "reports")
DATA = os.path.join(HOME, "data", "hbmep-processed")

RAT_TOML = os.path.join(CONFIG, "J_RCML.toml")
RAT_DATA = os.path.join(DATA, "rat", "J_RCML", "data.csv")
