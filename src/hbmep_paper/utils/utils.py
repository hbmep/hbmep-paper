import os
import glob
import itertools
import tomllib
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Optional

import mat73
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model.utils import Site as site
from hbmep.utils import timing
from hbmep.utils.constants import (
    RECRUITMENT_CURVES,
    MCMC_NC,
    DIAGNOSTICS_CSV,
    LOO_CSV,
    WAIC_CSV
)

from hbmep_paper.utils.constants import AUC_MAP

logger = logging.getLogger(__name__)


def _clean_subdural_epidural_dataset(df: pd.DataFrame):
    """
    Clean human data
    """
    # muscles = [
    #     "Trapezius", "Deltoid", "Biceps", "Triceps", "ECR", "FCR", "APB", "ADM", "TA", "EDB", "AH"
    # ]
    muscles = ["Trapezius", "Deltoid", "Biceps", "Triceps", "APB", "ADM"]

    # df.drop(columns=["sc_electrode"], axis=1, inplace=True)

    # experiment = [
    #     "sc_laterality",
    #     "sc_count",
    #     "sc_polarity",
    #     "sc_electrode_configuration",
    #     "sc_electrode_type",
    #     "sc_iti"
    # ]
    # subset = ["sc_cluster_as", "sc_current", "sc_level", "sc_cluster_as"]
    # # subset += [AUC_MAP["L" + muscle] for muscle in muscles]
    # # subset += [AUC_MAP["R" + muscle] for muscle in muscles]
    # dropna_subset = subset + experiment
    # df = df.dropna(subset=dropna_subset, axis="rows", how="any").copy()

    # Keep `mode` as `research_scs`
    df = df[df["mode"]=="research_scs"].copy()

    ind = df["participant"].isin(["scapptio017"])
    df = df[ind].reset_index(drop=True).copy()

    # ## Experiment filters
    # # Keep `sc_count` equal to 3
    # df = df[(df.sc_count.isin([3]))].copy()
    # Keep `sc_electrode_type` as `handheld`
    df = df[(df.sc_electrode_type.isin(["handheld"]))].copy()
    # Keep `sc_electrode_configuration` as `RC`
    df = df[(df.sc_electrode_configuration.isin(["RC"]))].copy()

    for muscle in muscles:
        df[muscle] = \
            df.apply(
                lambda x: x[AUC_MAP["R" + muscle]],
                axis=1
            )

    df["sc_current"] = df["sc_current"].apply(lambda x: x * 1000)

    df.reset_index(drop=True, inplace=True)
    return df


def _clean_intraoperative_data(df: pd.DataFrame, sc_approach: str):
    """
    Clean human data
    """
    # Validate input
    assert sc_approach in ["anterior", "posterior"]

    # muscles = [
    #     "Trapezius", "Deltoid", "Biceps", "Triceps", "ECR", "FCR", "APB", "ADM", "TA", "EDB", "AH"
    # ]
    muscles = ["Trapezius", "Deltoid", "Biceps", "Triceps", "APB", "ADM"]

    # `sc_electrode` and `sc_electrode_type` are in 1-1 relationship
    # # Can be verified using
    # df.groupby("sc_electrode")["sc_electrode_type"].apply(lambda x: x.nunique() == 1).all()
    # df.groupby("sc_electrode_type")["sc_electrode"].apply(lambda x: x.nunique() == 1).all()
    # We can drop either one
    df.drop(columns=["sc_electrode"], axis=1, inplace=True)

    experiment = [
        "sc_laterality",
        "sc_count",
        "sc_polarity",
        "sc_electrode_configuration",
        "sc_electrode_type",
        "sc_iti"
    ]
    subset = ["sc_cluster_as", "sc_current", "sc_level", "sc_cluster_as"]
    # subset += [AUC_MAP["L" + muscle] for muscle in muscles]
    # subset += [AUC_MAP["R" + muscle] for muscle in muscles]
    dropna_subset = subset + experiment
    df = df.dropna(subset=dropna_subset, axis="rows", how="any").copy()

    # Keep `mode` as `research_scs`
    df = df[df["mode"]=="research_scs"].copy()

    # Keep `sc_depth` as `epidural`
    df = df[(df.sc_depth.isin(["epidural"]))].copy()

    # # Both `reject_research_scs03` and `reject_research_scs14` must be simultaneously False
    # df = df[(df.reject_research_scs03)==False & (df.reject_research_scs14==False)].copy()

    # Filter by sc_approach
    df = df[df.sc_approach.isin([sc_approach])].copy()

    # Remove rows with `sc_laterality` not in ["L", "R", "M"] # RM?
    df = df[df.sc_laterality.isin(["L", "R", "M"])].copy()

    ## Experiment filters
    # Keep `sc_count` equal to 3
    df = df[(df.sc_count.isin([3]))].copy()
    # Keep `sc_electrode_type` as `handheld`
    df = df[(df.sc_electrode_type.isin(["handheld"]))].copy()
    # Keep `sc_electrode_configuration` as `RC`
    df = df[(df.sc_electrode_configuration.isin(["RC"]))].copy()

    keep_combinations = [
        ("cornptio001", "C7", "R"),
        ("cornptio003", "C7", "L"),
        ("cornptio004", "C7", "R"),
        ("cornptio007", "C6", "R"),
        ("cornptio008", "C8", "L"),
        ("cornptio010", "C6", "L"),
        ("cornptio011", "C7", "R"),
        ("cornptio012", "C8", "L"),
        ("cornptio013", "C7", "L"),
        ("cornptio014", "C7", "L"),
        ("cornptio015", "T1", "L"),
        ("cornptio017", "C4", "L"),
        ("scapptio001", "C8", "L")
    ]

    LATERALITY_MAP = {c[0]: c[2] for c in keep_combinations}

    keep_combinations = \
        keep_combinations + [(c[0], c[1], "M") for c in keep_combinations]

    idx = df.apply(lambda x: (x.participant, x.sc_level, x.sc_laterality), axis=1).isin(keep_combinations)
    df = df[idx].copy()

    # cornptio004 Remove `sc_cluster_as`
    remove_combinations = \
        [
            ("cornptio001", "C7", "M", 4),
            ("cornptio004", "C7", "M", 11),
            ("cornptio014", "C7", "M", 11),
            ("cornptio003", "C7", "M", 17),
            ("cornptio003", "C7", "M", 18),
            ("cornptio007", "C6", "R", 10),
            ("cornptio007", "C6", "M", 7)
        ]
    idx = \
        df.apply(lambda x: (x.participant, x.sc_level, x.sc_laterality, x.sc_cluster_as), axis=1) \
        .isin(remove_combinations)
    df = df[~idx].copy()

    df["participant"] = df["participant"]
    df["sc_current"] = df["sc_current"]
    df["sc_level"] = df["sc_level"]
    df["sc_laterality"] = df["sc_laterality"].apply(lambda x: x if x == "M" else "L")

    for muscle in muscles:
        df[muscle] = \
            df.apply(
                lambda x: x[AUC_MAP["L" + muscle]] \
                if LATERALITY_MAP[x.participant] == "L"
                else x[AUC_MAP["R" + muscle]],
                axis=1
            )

    df["sc_current"] = df["sc_current"].apply(lambda x: x * 1000)

    df.reset_index(drop=True, inplace=True)
    return df


@timing
def load_intraoperative_data(
    dir: Path,
    sc_approach: str = "posterior",
    subdural_epidural_only: bool = False
):
    DATASET_REAL = 'real_data/dnc_info_2022-05-26.parquet'
    df = pd.read_parquet(os.path.join(dir, DATASET_REAL))
    if subdural_epidural_only:
        df = _clean_subdural_epidural_dataset(df)
    else:
        df = _clean_intraoperative_data(df, sc_approach)
    return df


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
def simulate(
    model: BaseModel,
    n_subject=3,
    n_feature0=2,
    n_repeats=10,
    downsample_rate: int = 1
):
    n_features = [n_subject, n_feature0]
    combinations = itertools.product(*[range(i) for i in n_features])
    combinations = list(combinations)
    combinations = sorted(combinations)

    logger.info("Simulating data ...")
    df = pd.DataFrame(combinations, columns=model.combination_columns)
    x_space = np.arange(0, 360, 4)

    df[model.intensity] = df.apply(lambda _: x_space, axis=1)
    df = df.explode(column=model.intensity).reset_index(drop=True).copy()
    df[model.intensity] = df[model.intensity].astype(float)

    posterior_samples = model.predict(df=df, num_samples=n_repeats)
    posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}

    # x_space = x_space[1:] ?? Remove 0 from x-space?
    x_space_downsampled = x_space[::downsample_rate]
    ind = df[model.intensity].isin(x_space_downsampled)
    df = df[ind].reset_index(drop=True).copy()

    for var in [site.obs, site.mu, site.beta]:
        posterior_samples[var] = posterior_samples[var][:, ind, :]

    return df, posterior_samples


@timing
def run_experiment(
    config: Config,
    models: list[BaseModel],
    df: pd.DataFrame,
    posterior_samples_true: dict
):
    obs_true = posterior_samples_true[site.obs]
    n_repeats = obs_true.shape[0]
    n_models = len(models)

    baseline = BaseModel(config=config)
    df, encoder_dict = baseline.load(df=df)

    combinations = baseline._make_combinations(df=df, columns=baseline.combination_columns)
    combinations = [[slice(None)] + list(c) for c in combinations]
    combinations = [tuple(c[::-1]) for c in combinations]

    evaluation = []
    evaluation_path = os.path.join(baseline.build_dir, "evaluation.csv")
    columns = ["run", "model", "mse", "mae"]

    for i in range(n_repeats):
        a_true = np.array(posterior_samples_true[site.a])       # n_repeats x ... x n_muscles
        a_true = a_true[i, ...]      # ... x n_muscles
        a_true = np.array([a_true[c] for c in combinations])    # n_combinations x n_muscles
        a_true = a_true.reshape(-1, )

        for j, m in enumerate(models):
            logger.info("\n\n")
            logger.info(f" Experiment: {i + 1}/{n_repeats}, Model: {j + 1}/{n_models} ({m.LINK}) ".center(20, "="))
            model = m(config=config)
            df[model.response] = obs_true[i, ...]
            mcmc, posterior_samples = model.run_inference(df=df)

            a = np.array(posterior_samples[site.a])     # n_posterior_samples x ... x n_muscles
            a = a.mean(axis=0)      # ... x n_muscles
            a = np.array([a[c] for c in combinations])      # n_combinations x n_muscles
            a = a.reshape(-1, )

            # Evaluate
            mse = mean_squared_error(y_true=a_true, y_pred=a)
            mae = mean_absolute_error(y_true=a_true, y_pred=a)

            row = [i + 1, m.LINK, mse, mae]
            evaluation.append(row)
            logger.info(
                f"Run: {i + 1}/{n_repeats}, Model: {j + 1}/{n_models} ({m.LINK}), MSE: {mse}, MAE: {mae}"
            )

            # if not (i + 1) % 10:
            if True:
                evaluation_df = pd.DataFrame(evaluation, columns=columns)
                evaluation_df.to_csv(evaluation_path, index=False)

                model.build_dir = os.path.join(baseline.build_dir, f"run{i + 1}/{m.LINK}")
                model._make_dir(model.build_dir)

                model.recruitment_curves_path = os.path.join(model.build_dir, RECRUITMENT_CURVES)
                model.mcmc_path = os.path.join(model.build_dir, MCMC_NC)
                model.diagnostics_path = os.path.join(model.build_dir, DIAGNOSTICS_CSV)
                model.loo_path = os.path.join(model.build_dir, LOO_CSV)
                model.waic_path = os.path.join(model.build_dir, WAIC_CSV)

                logger.info(f"Saving artefacts to {model.build_dir}")
                model.render_recruitment_curves(df=df, encoder_dict=encoder_dict, posterior_samples=posterior_samples)
                model.save(mcmc=mcmc)

    evaluation_df = pd.DataFrame(evaluation, columns=columns)
    evaluation_df.to_csv(evaluation_path, index=False)
    return
