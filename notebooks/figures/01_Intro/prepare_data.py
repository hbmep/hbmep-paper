import os
import logging

import pandas as pd
import numpy as np

from hbmep.model import BaseModel
from hbmep.config import Config

from hbmep_paper.utils import setup_logging

logger = logging.getLogger(__name__)

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/rats/J_RCML_000.toml"
DATA_PATH = "/home/vishu/data/hbmep-processed/J_RCML_000/data.csv"
MAT_PATH = "/home/vishu/data/hbmep-processed/J_RCML_000/mat.npy"
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/figures/01_Intro"


def main():
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = BUILD_DIR

    model = BaseModel(config=config)
    model._make_dir(model.build_dir)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    c0, m0, const0 = ("amap08", "C6M-C7L"), "LECR", 1.8
    c1, m1, const1 = ("amap08", "C7M-C8M"), "LFCR", 1.

    df = pd.read_csv(DATA_PATH)
    mat = np.load(MAT_PATH)

    ind = (
        df[model._features[0]]
        .apply(tuple, axis=1)
        .isin([c0, c1])
    )
    df = df[ind].reset_index(drop=True).copy()
    mat = mat[ind, ...]
    logger.info(df.shape)
    logger.info(mat.shape)

    df = df.sort_values(by=[model.intensity]).copy()
    mat = mat[df.index.values]
    df = df.reset_index(drop=True).copy()

    ind0 = (
        df[model._features[0]]
        .apply(tuple, axis=1)
        .isin([c0])
    )
    ind1 = (
        df[model._features[0]]
        .apply(tuple, axis=1)
        .isin([c1])
    )
    assert (
        (df[ind0].reset_index(drop=True)[model.intensity] ==
        df[ind1].reset_index(drop=True)[model.intensity]).all()
    )

    df0 = df[ind0].reset_index(drop=True).copy()
    df1 = df[ind1].reset_index(drop=True).copy()
    df0[m1] = df1[m1].values
    df = df0.copy()
    df = df.rename(columns={
        m0: "APB", m1: "ADM"
    })
    df = df[[model.intensity, *model._features[0], "APB", "ADM"]].copy()

    m0_ind = [i for i, _response in enumerate(model.mep_data["mep_response"]) if _response == m0][0]
    m1_ind = [i for i, _response in enumerate(model.mep_data["mep_response"]) if _response == m1][0]
    mat = np.concatenate([mat[ind0, ..., m0_ind][..., None], mat[ind1, ..., m1_ind][..., None]], axis=-1)
    logger.info(df.shape)
    logger.info(mat.shape)
    logger.info(f"\n{df.to_string()}")

    model.mep_data["mep_response"] = ["APB", "ADM"]
    model.response = ["ADM", "APB"]
    df[model.response[0]] = df[model.response[0]] / const0
    df[model.response[1]] = df[model.response[1]] / const1
    df[model.intensity] = df[model.intensity] / 2.
    df_, encoder_dict = model.load(df=df)
    model.plot(df=df_, encoder_dict=encoder_dict, mep_matrix=mat)

    dest = os.path.join(model.build_dir, "data.csv")
    df.to_csv(dest, index=False)
    logger.info(f"Saved data to {dest}")

    dest = os.path.join(model.build_dir, "mat.npy")
    np.save(dest, mat)
    logger.info(f"Saved mat to {dest}")

    return


if __name__ == "__main__":
    main()
