import glob
import itertools
import tomllib
import logging
from tqdm import tqdm
from pathlib import Path

import mat73
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from hbmep.model import Baseline
from hbmep.model.utils import Site as site
from hbmep.utils import timing

logger = logging.getLogger(__name__)


@timing
def load_rats_data(
    dir: Path,
    subdir_pattern: list[str] = ["*L_CIRC*"],
    subjects: list[int] = range(1, 9)
):
    df = None

    for p in tqdm(subjects):
        subject = f"amap{p:02}"

        for pattern in subdir_pattern:
            PREFIX = f"{dir}/{subject}/{pattern}"

            subdirs = glob.glob(PREFIX)
            subdirs = sorted(subdirs)

            for subdir in subdirs:

                fpath = glob.glob(f"{subdir}/*auc_table.csv")[0]
                temp_df = pd.read_csv(fpath)

                fpath = glob.glob(f"{subdir}/*ep_matrix.mat")[0]
                data_dict = mat73.loadmat(fpath)

                temp_mat = data_dict["ep_sliced"]

                fpath = glob.glob(f"{subdir}/*cfg_proc.toml")[0]
                with open(fpath, "rb") as f:
                    cfg_proc = tomllib.load(f)

                fpath = glob.glob(f"{subdir}/*cfg_data.toml")[0]
                with open(fpath, "rb") as f:
                    cfg_data = tomllib.load(f)

                temp_df["participant"] = subject
                temp_df["subdir_pattern"] = pattern

                # Rename columns to actual muscle names
                muscles = cfg_data["muscles_emg"]
                muscles_map = {
                    f"auc_{i + 1}": m for i, m in enumerate(muscles)
                }
                temp_df = temp_df.rename(columns=muscles_map).copy()

                # Reorder MEP matrix
                temp_mat = temp_mat[..., np.argsort(muscles)]

                if df is None:
                    df = temp_df.copy()
                    mat = temp_mat

                    time = data_dict["t_sliced"]
                    auc_window = cfg_proc["auc"]["t_slice_minmax"]
                    muscles_sorted = sorted(cfg_data["muscles_emg"])

                    assert len(set(muscles_sorted)) == len(muscles_sorted)
                    continue

                assert (data_dict["t_sliced"] == time).all()
                assert cfg_proc["auc"]["t_slice_minmax"] == auc_window
                assert set(cfg_data["muscles_emg"]) == set(muscles_sorted)

                df = pd.concat([df, temp_df], ignore_index=True).copy()
                mat = np.vstack((mat, temp_mat))

    # Rename df response columns to auc_i
    muscles_map = {
        m: f"auc_{i + 1}" for i, m in enumerate(muscles_sorted)
    }
    df = df.rename(columns=muscles_map).copy()
    df.reset_index(drop=True, inplace=True)

    muscles_map = {
        v: u for u, v in muscles_map.items()
    }
    return df, mat, time, auc_window, muscles_map


@timing
def simulate(model: Baseline):
    x_space = np.arange(0, 360, 4)

    n_subject = 3
    n_features = [n_subject]
    n_features += jax.random.choice(model.rng_key, jnp.array([2, 3, 4]), shape=(model.n_features,)).tolist()
    n_features[-1] = 10

    combinations = itertools.product(*[range(i) for i in n_features])
    combinations = list(combinations)
    combinations = sorted(combinations)

    logger.info("Simulating data ...")
    df = pd.DataFrame(combinations, columns=model.combination_columns)
    df[model.intensity] = df.apply(lambda _: x_space, axis=1)
    df = df.explode(column=model.intensity).reset_index(drop=True).copy()
    df[model.intensity] = df[model.intensity].astype(float)

    pred = model.predict(df=df)
    obs = pred[site.obs]

    df[model.response] = obs[0, ...]
    return df
