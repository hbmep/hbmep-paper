DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"
MAT_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.npy"
TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/tms/config.toml"

import os
import numpy as np
import pandas as pd
from hbmep.config import Config
from hbmep.model import BaseModel
import hbmep as mep
import hbmep.functional as F

from hbmep_paper.utils import setup_logging

def main():
    config = Config(TOML_PATH)
    config.FEATURES = ["participant", "participant_condition"]
    config.BUILD_DIR = "/home/vishu/testing"
    config.RESPONSE = config.RESPONSE[0:1]
    # config.FEATURES = []
    setup_logging(config.BUILD_DIR, "logs")
    model = BaseModel(config)


    df = pd.read_csv(DATA_PATH)
    df, encoder_dict = model.load(df)
    # print(encoder_dict.keys())
    # print(type(encoder_dict.keys()))
    # print(type(encoder_dict))
    # print(model._features)
    # print(model.features)
    # print(model.regressors)
    # print(model.response)
    print(df[model.features].to_numpy().T.shape)

    ind = df[model.features[0]] == 0
    # print(ind.sum())

    mat = np.load(MAT_PATH)
    mep_time_range = [-0.15, 0.15]
    model.plot(df, mep_matrix=mat)
    return
    time = np.linspace(*mep_time_range, mat.shape[1], mat.shape[1])
    is_within_mep_window = (
        (time >= mep_time_range[0]) & (time <= mep_time_range[1])
    )
    mat = mat[ind, ...]
    max_amplitude = mat[..., is_within_mep_window, 0].max()

    mep_size_window = [0.015, 0.075]
    # import matplotlib.pyplot as plt

    # nrows, ncols = 1, 1
    # fig, axes = plt.subplots(
    #     nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 3), squeeze=False, constrained_layout=True
    # )
    # ax = axes[0, 0]
    # ax = model.mep_plot(
    #     ax,
    #     # (mat[..., 0] / max_amplitude) * (5),
    #     (mat[..., 0] / max_amplitude) * 5,
    #     df[model.intensity].values[ind],
    #     time=np.linspace(*mep_time_range, mat.shape[1]),
    #     color="purple",
    #     alpha=.4
    # )
    # ax.set_ylim(bottom=-0.001, top=mep_size_window[1] + (mep_size_window[0] - (-0.001)))
    # dest = os.path.join(config.BUILD_DIR, "test.png")
    # fig.savefig(dest)
    # print(f"saved to {dest}")



if __name__ == "__main__":
    main()