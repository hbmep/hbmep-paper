import os

import pandas as pd
import numpy as np

from hbmep.model import BaseModel


def prepare_data(config, data_path, mat_path):
    baseline = BaseModel(config=config)
    baseline._make_dir(baseline.build_dir)

    DF = pd.read_csv(data_path)
    MAT = np.load(mat_path)
    mep_response = ['LADM', 'LBiceps', 'LBicepsFemoris', 'LDeltoid', 'LECR', 'LFCR', 'LTriceps', 'RBiceps']

    DF = DF.sort_values(by=[baseline.intensity]).copy()
    MAT = MAT[DF.index.values]
    DF = DF.reset_index(drop=True).copy()

    c, muscle, const = ("amap08", "-C5M"), "LADM", 2.2
    ind = DF[baseline.features].apply(tuple, axis=1).isin([c])
    temp_DF = DF[ind].reset_index(drop=True).copy()
    df = pd.DataFrame(
        temp_DF[baseline.intensity],
        columns=[baseline.intensity]
    )
    df[baseline.response[0]] = temp_DF[muscle] / const

    mep_response_ind = [i for i, _response in enumerate(mep_response) if _response == muscle][0]
    mat = MAT[ind, :, mep_response_ind][..., None]

    c, muscle, const =("amap08", "C6M-C7L"), "LECR", 1
    ind = DF[baseline.features].apply(tuple, axis=1).isin([c])
    temp_DF = DF[ind].reset_index(drop=True).copy()
    df[baseline.response[1]] = temp_DF[muscle] / const

    mep_response_ind = [i for i, _response in enumerate(mep_response) if _response == muscle][0]
    mat = np.concatenate([
        mat,
        MAT[ind, :, mep_response_ind][..., None]
    ], axis=-1)

    df[baseline.features[0]] = 0
    df[baseline.features[1]] = 0

    dest = os.path.join(baseline.build_dir, "mat.npy")
    np.save(dest, mat)
    return df, mat
