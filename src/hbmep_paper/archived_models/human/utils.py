import os
import logging
from typing import Optional
from pathlib import Path

import pandas as pd

from hbmep.dataset import Dataset
from hbmep.utils import timing
from hbmep_paper.utils.constants import (
    AUC_MAP
)

logger = logging.getLogger(__name__)


@timing
def load_data(
    dir: Path,
    sc_approach: str = "posterior"
):
    DATASET_REAL = 'real_data/dnc_info_2022-05-26.parquet'
    df = pd.read_parquet(os.path.join(dir, DATASET_REAL))
    df = _clean_human_data(df, sc_approach)
    return df


def _clean_human_data(df: pd.DataFrame, sc_approach: str):
    """
    Clean human data
    """
    # Validate input
    assert sc_approach in ["anterior", "posterior"]

    muscles = [
        "Trapezius", "Deltoid", "Biceps", "Triceps", "ECR", "FCR", "APB", "ADM", "TA", "EDB", "AH"
    ]

    # muscles = [
    #     "Triceps"
    # ]

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