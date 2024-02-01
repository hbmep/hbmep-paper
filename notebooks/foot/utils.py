import os
import glob
import pickle
import tomllib
import logging

import arviz as az
import pandas as pd
from scipy.io import loadmat

logger = logging.getLogger(__name__)
MUSCLES = ["ADM", "APB", "Biceps", "ECR", "FCR", "Triceps"]
PKPK_MUSCLES = [f"PKPK_{m}" for m in MUSCLES]
AUC_MUSCLES = [f"AUC_{m}" for m in MUSCLES]


def process(df, mat, cfg):
    cfg_muscles = cfg["st"]["channel"]
    map = {
        f"pkpk_{i + 1}": f"pkpk_{m}"
        for i, m in enumerate(cfg_muscles)
    }
    df = df.rename(columns=map).copy()
    map = {
        f"auc_{i + 1}": f"auc_{m}"
        for i, m in enumerate(cfg_muscles)
    }
    df = df.rename(columns=map).copy()

    assert df["target_muscle"].unique().shape == (1,)
    side = df["target_muscle"].unique()[0][0]
    sided_MUSCLES = [side + m for m in MUSCLES]
    pkpk_sided_muscles = [f"pkpk_{m}" for m in sided_MUSCLES]
    auc_sided_muscles = [f"auc_{m}" for m in sided_MUSCLES]
    assert set(pkpk_sided_muscles) <= set(df.columns)
    assert set(auc_sided_muscles) <= set(df.columns)

    df[PKPK_MUSCLES] = df[pkpk_sided_muscles]
    df[AUC_MUSCLES] = df[auc_sided_muscles]

    d = {m: i for i, m in enumerate(cfg_muscles)}
    ind = [d[m] for m in sided_MUSCLES]
    mat = mat[..., ind]

    return df, mat


def read_data(subdir):
    pattern = os.path.join(subdir, "*REC_table.csv")
    logger.debug(f"table pattern: {pattern}")
    src = glob.glob(pattern)
    logger.debug(f"table: {src}")
    assert len(src) == 1
    src = src[0]
    df = pd.read_csv(src)

    pattern = os.path.join(subdir, "*ep_matrix.mat")
    src = glob.glob(pattern)
    assert len(src) == 1
    src = src[0]
    data_dict = loadmat(src)
    mat = data_dict["ep_sliced"]

    pattern = os.path.join(subdir, "*cfg_proc.toml")
    src = glob.glob(pattern)
    assert len(src) == 1
    src = src[0]
    with open(src, "rb") as f:
        cfg_proc = tomllib.load(f)

    # Fix TMS stim_type
    ind = df["stim_type"] == "TMS"
    logger.debug(f"Fixing TMS stim_type: {ind.sum()}")
    df = df[ind].reset_index(drop=True).copy()
    mat = mat[ind, ...]

    df, mat = process(df, mat, cfg_proc)
    assert set(PKPK_MUSCLES) <= set(df.columns)
    assert set(AUC_MUSCLES) <= set(df.columns)
    assert mat.shape[-1] == len(MUSCLES)
    logger.info(df["participant"].unique())
    assert df["participant"].unique().shape == (1,)
    return df, mat


def save(model, mcmc, posterior_samples):
    numpyro_data = az.from_numpyro(mcmc)
    logger.info("Evaluating model ...")
    score = az.loo(numpyro_data)
    logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
    score = az.waic(numpyro_data)
    logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")

    # Save posterior
    dest = os.path.join(model.build_dir, "inference.pkl")
    with open(dest, "wb") as f:
        pickle.dump((model, mcmc, posterior_samples), f)
    logger.info(dest)

    dest = os.path.join(model.build_dir, "numpyro_data.nc")
    az.to_netcdf(numpyro_data, dest)
    logger.info(dest)
