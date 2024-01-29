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
FEATURES = ["participant",]
RESPONSE = ["Biceps", "ECR"]
MEP_RESPONSE = ['LADM', 'LBiceps', 'LBicepsFemoris', 'LDeltoid', 'LECR', 'LFCR', 'LTriceps', 'RBiceps']
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/figures/01_Intro"


def main():
    config = Config(toml_path=TOML_PATH)
    config.BUILD_DIR = BUILD_DIR
    config.RESPONSE = RESPONSE

    model = BaseModel(config=config)
    model._make_dir(model.build_dir)
    setup_logging(
        dir=model.build_dir,
        fname=os.path.basename(__file__)
    )

    df = pd.read_csv(DATA_PATH)
    mat = np.load(MAT_PATH)
    mep_response = MEP_RESPONSE

    df = df.sort_values(by=[model.intensity]).copy()
    mat = mat[df.index.values]
    df = df.reset_index(drop=True).copy()

    DF = df.copy()
    MAT = mat.copy()
    c, muscle, const = ("amap08", "-C5M"), "LADM", 2.2
    ind = DF[model.features].apply(tuple, axis=1).isin([c])
    temp_DF = DF[ind].reset_index(drop=True).copy()
    df = pd.DataFrame(
        temp_DF[model.intensity],
        columns=[model.intensity]
    )
    df[model.response[0]] = temp_DF[muscle] / const

    mep_response_ind = [i for i, _response in enumerate(mep_response) if _response == muscle][0]
    mat = MAT[ind, :, mep_response_ind][..., None]

    c, muscle, const =("amap08", "C6M-C7L"), "LECR", 1
    ind = DF[model.features].apply(tuple, axis=1).isin([c])
    temp_DF = DF[ind].reset_index(drop=True).copy()
    df[model.response[1]] = temp_DF[muscle] / const

    mep_response_ind = [i for i, _response in enumerate(mep_response) if _response == muscle][0]
    mat = np.concatenate([
        mat,
        MAT[ind, :, mep_response_ind][..., None]
    ], axis=-1)

    df[model.features[0]] = 0
    df[model.features[1]] = 0

    df = df[[model.intensity, *model.response, model.features[0]]].copy()
    df = df.rename(columns={
        "Biceps": "APB", "ECR": "ADM"
    })
    dest = os.path.join(model.build_dir, "data.csv")
    df.to_csv(dest, index=False)
    logger.info(f"Saved data to {dest}")

    dest = os.path.join(model.build_dir, "mat.npy")
    np.save(dest, mat)
    logger.info(f"Saved mat to {dest}")

    return


if __name__ == "__main__":
    main()
