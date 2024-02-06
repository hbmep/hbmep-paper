import os
import glob
import logging

import pandas as pd
import numpy as np

from utils import read_data

logger = logging.getLogger(__name__)

DATA_DIR = "/home/vishu/data/raw/foot"


if __name__ == "__main__":
    src = os.path.join(DATA_DIR, "*")
    visits = glob.glob(src)
    visits = [os.path.basename(v) for v in visits]

    d = {}
    for visit in visits:
        src = os.path.join(DATA_DIR, visit, "*")
        participants = glob.glob(src)
        participants = [os.path.basename(p) for p in participants]
        d[visit] = participants

    df = None
    mat = None
    for visit in visits:
        for participant in d[visit]:
            subdir = os.path.join(DATA_DIR, visit, participant)
            temp_df, temp_mat = read_data(subdir)
            temp_df["participant"] = participant
            temp_df["visit"] = visit
            if df is None:
                df = temp_df
                mat = temp_mat
            else:
                df = pd.concat([df, temp_df], axis=0)
                df = df.reset_index(drop=True).copy()
                mat = np.concatenate([mat, temp_mat], axis=0)

    dest = "/home/vishu/data/hbmep-processed/foot/data.csv"
    df.to_csv(dest, index=False)

    dest = "/home/vishu/data/hbmep-processed/foot/mat.npy"
    np.save(dest, mat)
